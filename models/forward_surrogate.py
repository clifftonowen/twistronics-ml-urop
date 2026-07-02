"""Forward surrogate (Stage 2): predict the CD spectrum from design parameters.

Learns X = (theta, thickness, gap, radius) -> CD spectrum over the 600-850 nm
grid with a small MLP, as a fast (sub-millisecond) stand-in for the RCWA solver
inside design loops.

Design notes
------------
- Trains on a processed dataset shard (X already standardized; Y = CD). Every
  design is an independent structure (Latin-hypercube), so a random row split is
  already a by-structure split; the held-out test set is RCWA ground truth.
- The dataset is small (v0 ~ 120 designs), so the model is deliberately small and
  regularized (weight decay + early stopping), and we report against a
  mean-spectrum baseline so "did it learn anything?" is unambiguous.
- The headline output is a LEARNING CURVE (val error vs training-set size): it
  tells us whether generating more (expensive) data would actually help, before
  we spend the compute.

Usage
-----
    python -m models.forward_surrogate --shard v0_n120
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
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_shard(shard: str, proc_dir: str = DEFAULT_PROC):
    """Return (X_std, Y_cd, wavelengths). X is already standardized in the shard."""
    data = np.load(os.path.join(proc_dir, f"{shard}.npz"))
    X = data["X"].astype(np.float32)          # (n, 4) standardized
    Y = data["Y"].astype(np.float32)          # (n, n_lambda) CD
    wl = data["wavelengths_nm"]
    return X, Y, wl


def split_indices(n: int, seed: int, fracs=(0.70, 0.15, 0.15)):
    """Shuffle once, then slice into train/val/test index arrays."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


class MLP(nn.Module):
    """Small fully-connected regressor: n_in -> hidden... -> n_out."""

    def __init__(self, n_in: int, n_out: int, hidden=(64, 64), p_drop: float = 0.0):
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


def _standardize_y(Y_train: np.ndarray):
    """Per-wavelength mean/std from the TRAIN split; std floored to avoid /0."""
    mean = Y_train.mean(axis=0, keepdims=True)
    std = Y_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def train_model(
    Xtr, Ytr, Xva, Yva,
    hidden=(64, 64), p_drop=0.0, lr=3e-3, weight_decay=1e-3,
    epochs=3000, patience=300, seed=0, verbose=False,
):
    """Train one MLP with early stopping on val MSE. Y is standardized internally;
    the returned model predicts in standardized-Y space (invert with y_mean/y_std).
    Returns (model, y_mean, y_std, history)."""
    set_seed(seed)
    y_mean, y_std = _standardize_y(Ytr)

    xtr = torch.from_numpy(Xtr)
    xva = torch.from_numpy(Xva)
    ytr = torch.from_numpy((Ytr - y_mean) / y_std)
    yva = torch.from_numpy((Yva - y_mean) / y_std)

    model = MLP(Xtr.shape[1], Ytr.shape[1], hidden=hidden, p_drop=p_drop)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v = loss_fn(model(xva), yva).item()
        history["train"].append(loss.item())
        history["val"].append(v)

        if v < best_val - 1e-6:
            best_val, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, y_mean, y_std, history


def predict(model, X, y_mean, y_std) -> np.ndarray:
    """Predict CD (original units) for standardized inputs X."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X.astype(np.float32))).numpy()
    return out * y_std + y_mean


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """RMSE / MAE in CD units, plus R^2 against the mean-spectrum predictor."""
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def baseline_metrics(Ytr, Yte) -> dict:
    """Predict the train-mean spectrum for every test design (sanity floor)."""
    pred = np.repeat(Ytr.mean(axis=0, keepdims=True), Yte.shape[0], axis=0)
    return metrics(pred, Yte)


def learning_curve(X, Y, tr_idx, va_idx, sizes, n_repeats=3, seed=0, **train_kw):
    """Val RMSE as training-set size grows -- does more data help?

    For each size k, train n_repeats fresh models on random k-subsets of the
    train split and average val RMSE. Returns (sizes, mean_rmse, std_rmse)."""
    Xva, Yva = X[va_idx], Y[va_idx]
    rng = np.random.default_rng(seed)
    means, stds = [], []
    for k in sizes:
        vals = []
        for r in range(n_repeats):
            sub = rng.choice(tr_idx, size=k, replace=False)
            model, ym, ys, _ = train_model(
                X[sub], Y[sub], Xva, Yva, seed=seed + r, **train_kw
            )
            vals.append(metrics(predict(model, Xva, ym, ys), Yva)["rmse"])
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return list(sizes), means, stds


def make_plots(out_dir, history, lc, pred_te, Yte, wl, peak_idx):
    """Training curve, learning curve, and predicted-vs-true CD for sample designs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written = []

    # 1. training / val loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set(xlabel="epoch", ylabel="MSE (standardized Y)", title="Training curve", yscale="log")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out_dir, "training_curve.png"); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)

    # 2. learning curve
    sizes, means, stds = lc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3)
    ax.set(xlabel="# training designs", ylabel="val RMSE (CD)",
           title="Learning curve — does more data help?")
    ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out_dir, "learning_curve.png"); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)

    # 3. predicted vs true CD spectra for a few test designs
    m = min(4, pred_te.shape[0])
    fig, axes = plt.subplots(1, m, figsize=(4 * m, 3.4), squeeze=False)
    for j in range(m):
        ax = axes[0][j]
        ax.plot(wl, Yte[j], "o-", label="RCWA")
        ax.plot(wl, pred_te[j], "x--", label="surrogate")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="wavelength (nm)", ylim=(-1, 1))
        if j == 0:
            ax.set_ylabel("CD"); ax.legend(fontsize=8)
    fig.suptitle("Predicted vs true CD (test designs)")
    fig.tight_layout(); p = os.path.join(out_dir, "pred_vs_true.png"); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", default="v0_n120", help="processed shard name")
    p.add_argument("--proc-dir", default=DEFAULT_PROC)
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

    X, Y, wl = load_shard(args.shard, args.proc_dir)
    n = X.shape[0]
    tr, va, te = split_indices(n, args.seed)
    print(f"shard={args.shard} | n={n} (train {len(tr)} / val {len(va)} / test {len(te)}) "
          f"| X{tuple(X.shape)} -> CD{tuple(Y.shape)}")

    train_kw = dict(hidden=tuple(args.hidden), p_drop=args.dropout, lr=args.lr,
                    weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience)

    model, y_mean, y_std, history = train_model(X[tr], Y[tr], X[va], Y[va], seed=args.seed, **train_kw)

    m_train = metrics(predict(model, X[tr], y_mean, y_std), Y[tr])
    m_val = metrics(predict(model, X[va], y_mean, y_std), Y[va])
    m_test = metrics(predict(model, X[te], y_mean, y_std), Y[te])
    base_test = baseline_metrics(Y[tr], Y[te])
    # train fit is the key diagnostic: if the model can't even fit train, the
    # issue is optimization/capacity; if it fits train but not val, it's
    # data-limited (undersampled) -- more data (or smoother targets) would help.
    print(f"\nsurrogate  train: RMSE {m_train['rmse']:.4f}  MAE {m_train['mae']:.4f}  R2 {m_train['r2']:.3f}")
    print(f"surrogate  val : RMSE {m_val['rmse']:.4f}  MAE {m_val['mae']:.4f}  R2 {m_val['r2']:.3f}")
    print(f"surrogate  test: RMSE {m_test['rmse']:.4f}  MAE {m_test['mae']:.4f}  R2 {m_test['r2']:.3f}")
    print(f"mean-pred  test: RMSE {base_test['rmse']:.4f}  MAE {base_test['mae']:.4f}  "
          f"(surrogate should beat this)")

    sizes = sorted(set([max(10, len(tr) // 4), len(tr) // 2, 3 * len(tr) // 4, len(tr)]))
    lc = learning_curve(X, Y, tr, va, sizes, seed=args.seed, **train_kw)
    print("\nlearning curve (val RMSE vs # train designs):")
    for k, mu, sd in zip(*lc):
        print(f"  n={k:3d}: RMSE {mu:.4f} +/- {sd:.4f}")

    os.makedirs(args.out, exist_ok=True)
    pred_te = predict(model, X[te], y_mean, y_std)
    plots = make_plots(args.out, history, lc, pred_te, Y[te], wl, te)

    ckpt = os.path.join(args.out, f"forward_surrogate_{args.shard}.pt")
    torch.save(
        {"state_dict": model.state_dict(), "hidden": list(args.hidden),
         "y_mean": y_mean, "y_std": y_std, "n_in": X.shape[1], "n_out": Y.shape[1],
         "shard": args.shard, "seed": args.seed},
        ckpt,
    )
    report = {
        "shard": args.shard, "n": n, "split": [len(tr), len(va), len(te)],
        "hidden": list(args.hidden), "weight_decay": args.weight_decay,
        "val": m_val, "test": m_test, "baseline_test": base_test,
        "learning_curve": {"sizes": lc[0], "val_rmse": lc[1], "val_rmse_std": lc[2]},
    }
    with open(os.path.join(args.out, f"forward_surrogate_{args.shard}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nsaved model -> {ckpt}")
    for p in plots:
        print(f"  plot: {p}")
    return report


if __name__ == "__main__":
    main()
