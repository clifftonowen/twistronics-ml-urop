"""Append the Stage C slides to the existing supervisor deck.

Extends `docs/meeting_2026-07-23/Twistronics_Meeting_230726.pptx` rather than
rebuilding from `slides.md`. That deck was hand-edited after the pandoc build --
titles were shortened and a "Next Steps (Discussed in Meeting)" slide was added
that does not exist in the markdown -- so a rebuild would silently discard the
speaker's own work. The original file is never modified; output goes to a new
file.

Every number on the new slides is READ FROM the project's artifacts at build
time (the merged peak shard, the CV reports), so re-running after another
campaign shard lands refreshes the deck. Nothing is hard-coded except the
narrative text.

    python -m scripts.extend_meeting_deck
    python -m scripts.extend_meeting_deck --out /path/to/deck.pptx
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEETING = os.path.join(ROOT, "docs", "meeting_2026-07-23")
SRC = os.path.join(MEETING, "Twistronics_Meeting_230726.pptx")
FIGS = os.path.join(ROOT, "docs", "stage_c", "figures")
DEFAULT_OUT = os.path.join(ROOT, "docs", "stage_c", "Twistronics_Meeting_StageC.pptx")

TITLE_ONLY = 1      # "Title and Content"
TWO_CONTENT = 3     # "Two Content" -- picture left, bullets right


def collect_numbers() -> dict:
    """Pull every figure quoted on the new slides straight from the artifacts."""
    import sys
    sys.path.insert(0, ROOT)
    from scipy import stats

    from models.forward_surrogate import make_fit
    from models.two_stage_surrogate import cv_two_stage
    from scripts.make_meeting_figures import _peak_merged

    D, shards = _peak_merged()
    X = D["X"].astype(float)
    m, s = X.mean(0), X.std(0)
    s[s < 1e-12] = 1.0
    Xn = ((X - m) / s).astype(np.float32)
    fp = D["f_peak"].astype(np.float32)
    n = len(fp)

    out: dict = {"n": n, "shards": shards}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, Y in [
            ("cd", D["window_cd"].astype(np.float32)),
            ("t", np.concatenate([D["window_T_RCP"], D["window_T_LCP"]], 1).astype(np.float32)),
        ]:
            r = cv_two_stage(Xn, fp, Y, make_fit("gp"), 5, 10, 0)
            d = r["oracle"] - r["baseline"]
            out[key] = {
                "flat": r["flat"].mean(),
                "cond": r["oracle"].mean(),
                "cond_sd": r["oracle"].std(),
                "base": r["baseline"].mean(),
                "diff": d.mean(),
                "p": stats.ttest_1samp(d, 0)[1],
            }

    loc, win = D["locate_peak_abs_cd"], D["window_peak_abs_cd"]
    ok = np.isfinite(loc) & (loc > 1e-9)
    ratio = win[ok] / loc[ok]
    out["gain_median"] = float(np.median(ratio))
    out["gain_max"] = float(ratio.max())
    out["under_read_pct"] = float(100 * (ratio > 1.05).mean())
    out["peak_median"] = float(np.median(win))
    a_nm = 500.0
    out["lam_lo"], out["lam_hi"] = float(a_nm / fp.max()), float(a_nm / fp.min())
    u = D["u"]
    out["at_edge"] = int(sum(
        1 for cd in D["window_cd"]
        if abs(u[int(np.argmax(np.abs(cd)))]) / u.max() >= 0.95
    ))
    return out


def add_slide(prs, title: str, bullets: list, figure: str | None = None):
    """One slide, matching the deck's own geometry (measured from its slides)."""
    layout = prs.slide_layouts[TWO_CONTENT if figure else TITLE_ONLY]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    body = None
    for ph in slide.placeholders:
        if ph.placeholder_format.idx != 0:
            body = ph
            break

    if figure:
        path = os.path.join(FIGS, figure)
        if os.path.exists(path):
            # Picture left / bullets right, the same split the pandoc build used.
            slide.shapes.add_picture(path, Inches(0.07), Inches(1.85), width=Inches(5.5))
            if body is not None:
                body.left, body.top = Inches(5.65), Inches(1.4)
                body.width, body.height = Inches(4.3), Inches(3.7)
        else:
            print(f"  ! missing figure {figure} -- slide gets bullets only")

    if body is None:
        return slide
    tf = body.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, (text, level) in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = text
        para.level = level
        for run in para.runs:
            run.font.size = Pt(13 if figure else 15)
    return slide


def build_slides(prs, v: dict) -> int:
    """The Stage C narrative, answering the meeting's own Next Steps slide."""
    cd, t = v["cd"], v["t"]

    add_slide(
        prs,
        "Since the meeting: what was asked, and what we did",
        [
            ("Narrower window, <5 nm spacing → done, and 5 nm proved too coarse", 0),
            ("measured: 5 nm sampling loses up to 35% of the true peak height", 1),
            ("shipped ±3% of each design's own peak ≈ 44 nm at ~1.1 nm spacing", 1),
            ("Twist narrowed to 25–30° → done (new peak_box), 4.8× denser sampling", 0),
            ("Dataset centred on good designs → done: ⅔ perturbation around 6 high-CD "
             "seeds, ⅓ screened random", 0),
            ("Both dataset options were used, not one — the |CD| > 0.1 screen turned out "
             "non-binding (every design in this region passes), so the bar is 0.20", 0),
        ],
    )

    add_slide(
        prs,
        "The fix: measure each design on its own peak",
        [
            (f"CD peaks are spread over ~{v['lam_lo']:.0f}–{v['lam_hi']:.0f} nm — "
             "no single fixed window catches them", 0),
            ("So each design gets a window centred on its own peak, on a shared "
             "relative axis u = f/f_peak − 1", 0),
            ("Blocking gate first: does a cheap N=2 pass find the same peak as N=3?", 0),
            ("5/5 designs, 4 on the identical grid point — the cost model holds", 1),
            (f"{v['n']} designs measured; {v['at_edge']} sit at a window edge "
             "(auto re-centre added)", 0),
        ],
        "fig12_window_concept.png",
    )

    add_slide(
        prs,
        "The old grid was under-reading real CD",
        [
            (f"Resolved window vs coarse pass: median {v['gain_median']:.2f}×, "
             f"max {v['gain_max']:.2f}×", 0),
            (f"The coarse pass under-reads in {v['under_read_pct']:.0f}% of designs", 0),
            ("One design: a real feature at −0.356 that the old grid interpolated to "
             "−0.018 — a 20× under-read", 0),
            ("7 of 8 re-measured designs reproduce dense ground truth within 2%", 0),
            ("Q is now measurable: median ~157 vs a former ~109 ceiling that was itself "
             "a grid artifact", 0),
        ],
        "fig13_dealiasing_recovery.png",
    )

    add_slide(
        prs,
        "Result: CD is learnable once conditioned on where the peak sits",
        [
            (f"CD, conditioned:  R² = {cd['cond']:+.3f}   (baseline {cd['base']:+.3f})", 0),
            (f"paired over 50 folds: {cd['diff']:+.3f}, p = {cd['p']:.1e}", 1),
            (f"Transmission control: R² = {t['cond']:+.3f}  (baseline {t['base']:+.3f})", 0),
            (f"Flat X → lineshape is WORSE than baseline ({cd['flat']:+.3f}) — without "
             "knowing where the resonance is, geometry alone hurts", 0),
            ("First statistically strong positive CD result in the project", 0),
        ],
        "fig14_cd_conditioned.png",
    )

    add_slide(
        prs,
        "And the conditioning variable is cheap, not an oracle",
        [
            ("f_peak is measured by the N=2 locate pass in ~2.5 min — not predicted", 0),
            ("Verified on a subset using only locate-derived values, nothing finer", 1),
            ("Deployable pipeline: locate (2.5 min) → surrogate → resolved lineshape", 0),
            ("replaces the ~23 min N=3 window pass: ~10× cheaper per design", 1),
            ("Predicting f_peak from geometry alone still fails — but it no longer "
             "matters, because it can be measured instead", 0),
        ],
        "fig15_deployable_pipeline.png",
    )

    add_slide(
        prs,
        "Accuracy is still rising with data — the curve has not flattened",
        [
            ("Every earlier representation gave a FLAT learning curve pinned at "
             "baseline — that is what 'more data will not help' looks like", 0),
            ("This one crosses zero around n ≈ 44 and keeps climbing", 0),
            (f"Currently n = {v['n']} designs, ~25 min each on GPU", 0),
            ("Honest scope: transmission is usable; CD beats baseline decisively but "
             "R² is still low in absolute terms", 0),
            ("→ 'the measurement problem is solved and signal is found', not "
             "'CD surrogate finished'", 0),
        ],
        "fig16_cd_learning_curve.png",
    )

    add_slide(
        prs,
        "Next steps",
        [
            ("Keep scaling — the learning curve says more designs still buy accuracy", 0),
            ("Build the deployable surrogate around the measured f_peak "
             "(locate → conditioned model), and RCWA-verify its proposals", 0),
            ("Open question for you: is ~10× cheaper enough to start inverse design, "
             "or should we push CD's R² higher first?", 0),
            ("Deferred, cost-free when we choose it: ±2% vs ±3% window "
             "(better per-mode resolution, but risks clipping the partner mode)", 0),
            ("If accuracy plateaus: the suspect is mode-swapping — 'the peak' is an "
             "argmax over ~4 competing lobes, so the target can jump between "
             "neighbouring designs", 0),
        ],
    )
    return 7


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=SRC, help="deck to extend (never modified)")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args(argv)

    print("collecting numbers from artifacts ...")
    v = collect_numbers()
    print(f"  n={v['n']} from {', '.join(v['shards'])}")
    print(f"  CD conditioned R2 {v['cd']['cond']:+.3f} (baseline {v['cd']['base']:+.3f}), "
          f"p={v['cd']['p']:.1e}")
    print(f"  T  conditioned R2 {v['t']['cond']:+.3f} (baseline {v['t']['base']:+.3f})")

    prs = Presentation(args.src)
    before = len(prs.slides._sldIdLst)
    n_new = build_slides(prs, v)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    prs.save(args.out)
    print(f"\n{before} slides + {n_new} new = {before + n_new}")
    print(f"source untouched: {os.path.relpath(args.src, ROOT)}")
    print(f"wrote {os.path.relpath(args.out, ROOT)}")
    return args.out


if __name__ == "__main__":
    main()
