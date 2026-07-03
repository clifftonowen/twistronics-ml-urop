"""Forward surrogate (Stage 2): predict the optical response from design parameters.

Learns X = (theta, thickness, gap, radius) -> optical response with a fast model,
as a stand-in for the RCWA solver inside design loops.

Choose the target and the model:
  --target  cd | t_rcp | t_lcp | both_t | mean_t | delta_t
  --model   mlp | gp

Why the choice matters (measured on v0_n120, 120 designs)
---------------------------------------------------------
Transmission is smooth and learnable even at n=120 (GP R2 ~ 0.8 on the full
spectrum). CD is a small normalized difference of two nearly-equal transmissions
(catastrophic cancellation) and is NOT learnable at this sampling density /
uniform sampling (R2 ~ 0). So the working Stage-2 deliverable today is a
TRANSMISSION surrogate (`--target both_t --model gp`); a CD surrogate needs
high-CD-targeted data (see the data-campaign plan).

Every design is an independent structure (Latin-hypercube), so a random row split
is already a by-structure split; the held-out test set is RCWA ground truth. The
learning curve (val error vs training-set size) shows whether more data helps.

Usage
-----
    python -m models.forward_surrogate --target both_t --model gp --shard v0_n120
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_PROC = os.path.join(ROOT, "datasets", "processed")
DEFAULT_RAW = os.path.join(ROOT, "datasets", "raw")
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")

TARGETS = ("cd", "t_rcp", "t_lcp", "both_t", "mean_t", "delta_t")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(shard: str, target: str, proc_dir=DEFAULT_PROC, raw_dir=DEFAULT_RAW):
    """X (standardized inputs) and Y (chosen target), sharing design order."""
    proc = np.load(os.path.join(proc_dir, f"{shard}.npz"))
    raw = np.load(os.path.join(raw_dir, f"{shard}.npz"))
    X = proc["X"].astype(np.float32)          # (n, 4) standardized
    wl = proc["wavelengths_nm"]
    if target == "cd":
        Y = raw["CD"]
    elif target == "t_rcp":
        Y = raw["T_RCP"]
    elif target == "t_lcp":
        Y = raw["T_LCP"]
    elif target == "mean_t":
        Y = 0.5 * (raw["T_RCP"] + raw["T_LCP"])
    elif target == "both_t":
        Y = np.concatenate([raw["T_RCP"], raw["T_LCP"]], axis=1)  # (n, 2*n_lambda)
    elif target == "delta_t":
        # Unnormalized T_RCP - T_LCP, regressed directly (not via two separately
        # predicted T's, which v0 showed performs worse: R2=-0.04). Recovers CD
        # downstream via CD = delta_t / (T_RCP+T_LCP), using the (learnable)
        # T-sum -- avoids baking the 1/(T_RCP+T_LCP) noise-amplifying divide
        # into the training loss itself.
        Y = raw["T_RCP"] - raw["T_LCP"]
    else:
        raise ValueError(f"unknown target {target!r}; choose from {TARGETS}")
    return X, Y.astype(np.float32), wl


def split_indices(n: int, seed: int, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """RMSE / MAE in target units, plus R^2 against the per-output mean predictor."""
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def baseline_metrics(Ytr, Yte) -> dict:
    pred = np.repeat(Ytr.mean(axis=0, keepdims=True), Yte.shape[0], axis=0)
    return metrics(pred, Yte)


# --------------------------------------------------------------------------- #
# Models: a uniform fit(Xtr, Ytr, Xva, Yva) -> object with .predict(X)         #
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(64, 64), p_drop=0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPSurrogate:
    def __init__(self, model, y_mean, y_std, history=None):
        self.model, self.y_mean, self.y_std, self.history = model, y_mean, y_std, history

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(X.astype(np.float32))).numpy()
        return out * self.y_std + self.y_mean


def fit_mlp(Xtr, Ytr, Xva, Yva, hidden=(64, 64), p_drop=0.0, lr=3e-3,
            weight_decay=1e-3, epochs=3000, patience=300, seed=0):
    set_seed(seed)
    y_mean = Ytr.mean(0, keepdims=True).astype(np.float32)
    y_std = Ytr.std(0, keepdims=True).astype(np.float32)
    y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)

    xtr, xva = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    ytr = torch.from_numpy((Ytr - y_mean) / y_std)
    yva = torch.from_numpy((Yva - y_mean) / y_std)

    model = MLP(Xtr.shape[1], Ytr.shape[1], hidden=hidden, p_drop=p_drop)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    history = {"train": [], "val": []}
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(xtr), ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(xva), yva).item()
        history["train"].append(loss.item()); history["val"].append(v)
        if v < best_val - 1e-6:
            best_val, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return MLPSurrogate(model, y_mean, y_std, history)


class GPSurrogate:
    def __init__(self, gp):
        self.gp = gp

    def predict(self, X):
        pred = self.gp.predict(X)
        return pred[:, None] if pred.ndim == 1 else pred


def fit_gp(Xtr, Ytr, Xva=None, Yva=None, seed=0):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    kernel = C(1.0) * RBF(length_scale=np.ones(Xtr.shape[1])) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=seed)
    gp.fit(Xtr, Ytr)
    return GPSurrogate(gp)


def make_fit(model_name, **mlp_kw):
    if model_name == "mlp":
        return lambda Xtr, Ytr, Xva, Yva, seed=0: fit_mlp(Xtr, Ytr, Xva, Yva, seed=seed, **mlp_kw)
    if model_name == "gp":
        return lambda Xtr, Ytr, Xva, Yva, seed=0: fit_gp(Xtr, Ytr, seed=seed)
    raise ValueError(f"unknown model {model_name!r}")


def learning_curve(fit_fn, X, Y, tr_idx, va_idx, sizes, n_repeats=3, seed=0):
    Xva, Yva = X[va_idx], Y[va_idx]
    rng = np.random.default_rng(seed)
    means, stds = [], []
    for k in sizes:
        vals = []
        for r in range(n_repeats):
            sub = rng.choice(tr_idx, size=k, replace=False)
            sur = fit_fn(X[sub], Y[sub], Xva, Yva, seed=seed + r)
            vals.append(metrics(sur.predict(Xva), Yva)["rmse"])
        means.append(float(np.mean(vals))); stds.append(float(np.std(vals)))
    return list(sizes), means, stds


def make_plots(out_dir, tag, history, lc, pred_te, Yte, wl):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written = []
    if history is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["train"], label="train"); ax.plot(history["val"], label="val")
        ax.set(xlabel="epoch", ylabel="MSE (standardized)", title=f"Training curve ({tag})", yscale="log")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        p = os.path.join(out_dir, f"training_curve_{tag}.png"); fig.savefig(p, dpi=120); plt.close(fig)
        written.append(p)

    sizes, means, stds = lc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3)
    ax.set(xlabel="# training designs", ylabel="val RMSE", title=f"Learning curve ({tag})")
    ax.grid(alpha=0.3); fig.tight_layout()
    p = os.path.join(out_dir, f"learning_curve_{tag}.png"); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)

    m = min(4, pred_te.shape[0])
    n_lam = len(wl)
    fig, axes = plt.subplots(1, m, figsize=(4 * m, 3.4), squeeze=False)
    for j in range(m):
        ax = axes[0][j]
        # for both_t the first n_lam cols are T_RCP; plot that slice for readability
        ax.plot(wl, Yte[j, :n_lam], "o-", label="RCWA")
        ax.plot(wl, pred_te[j, :n_lam], "x--", label="surrogate")
        ax.set(xlabel="wavelength (nm)")
        if j == 0:
            ax.set_ylabel(tag); ax.legend(fontsize=8)
    fig.suptitle(f"Predicted vs true ({tag}, test designs)")
    fig.tight_layout()
    p = os.path.join(out_dir, f"pred_vs_true_{tag}.png"); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", default="v0_n120")
    p.add_argument("--target", choices=TARGETS, default="both_t")
    p.add_argument("--model", choices=("mlp", "gp"), default="gp")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=ARTIFACTS)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)

    X, Y, wl = load_dataset(args.shard, args.target)
    n = X.shape[0]
    tr, va, te = split_indices(n, args.seed)
    tag = f"{args.target}_{args.model}"
    print(f"shard={args.shard} target={args.target} model={args.model} | "
          f"n={n} (train {len(tr)}/val {len(va)}/test {len(te)}) | X{tuple(X.shape)} -> Y{tuple(Y.shape)}")

    fit_fn = make_fit(args.model, hidden=tuple(args.hidden), p_drop=args.dropout,
                      lr=args.lr, weight_decay=args.weight_decay,
                      epochs=args.epochs, patience=args.patience)

    sur = fit_fn(X[tr], Y[tr], X[va], Y[va], seed=args.seed)
    m_train = metrics(sur.predict(X[tr]), Y[tr])
    m_val = metrics(sur.predict(X[va]), Y[va])
    m_test = metrics(sur.predict(X[te]), Y[te])
    base = baseline_metrics(Y[tr], Y[te])
    print(f"\ntrain : RMSE {m_train['rmse']:.4f}  R2 {m_train['r2']:.3f}")
    print(f"val   : RMSE {m_val['rmse']:.4f}  R2 {m_val['r2']:.3f}")
    print(f"test  : RMSE {m_test['rmse']:.4f}  R2 {m_test['r2']:.3f}")
    print(f"mean-pred test: RMSE {base['rmse']:.4f}  R2 {base['r2']:.3f}  (surrogate should beat this)")

    sizes = sorted(set([max(10, len(tr) // 4), len(tr) // 2, 3 * len(tr) // 4, len(tr)]))
    lc = learning_curve(fit_fn, X, Y, tr, va, sizes, seed=args.seed)
    print("\nlearning curve (val RMSE vs # train designs):")
    for k, mu, sd in zip(*lc):
        print(f"  n={k:3d}: RMSE {mu:.4f} +/- {sd:.4f}")

    os.makedirs(args.out, exist_ok=True)
    history = getattr(sur, "history", None)
    plots = make_plots(args.out, tag, history, lc, sur.predict(X[te]), Y[te], wl)

    # Persist the fitted model so it can be reused for sub-second inference.
    if args.model == "gp":
        import joblib
        model_path = os.path.join(args.out, f"forward_surrogate_{tag}.joblib")
        joblib.dump({"gp": sur.gp, "target": args.target, "shard": args.shard}, model_path)
    else:
        model_path = os.path.join(args.out, f"forward_surrogate_{tag}.pt")
        torch.save(
            {"state_dict": sur.model.state_dict(), "hidden": list(args.hidden),
             "y_mean": sur.y_mean, "y_std": sur.y_std, "n_in": X.shape[1],
             "n_out": Y.shape[1], "target": args.target, "shard": args.shard},
            model_path,
        )

    report = {
        "shard": args.shard, "target": args.target, "model": args.model, "n": n,
        "split": [len(tr), len(va), len(te)],
        "train": m_train, "val": m_val, "test": m_test, "baseline_test": base,
        "learning_curve": {"sizes": lc[0], "val_rmse": lc[1], "val_rmse_std": lc[2]},
    }
    with open(os.path.join(args.out, f"forward_surrogate_{tag}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved model  -> {model_path}")
    print(f"saved report -> forward_surrogate_{tag}_report.json")
    for p in plots:
        print(f"  plot: {p}")
    return report


if __name__ == "__main__":
    main()
