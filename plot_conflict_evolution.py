"""
How the AIM policy evolves during training -- Uni-Mol vs GNN, across epochs.

Question: does pretraining change how conflict develops during optimization?

IMPORTANT -- what this figure can and cannot show
-------------------------------------------------
train.py never logs cos(g_i, g_j). The per-step gradient cosines exist only
inside _step_aim and are discarded, so there is NO measured conflict trace on
disk for any run. What history.json does record every epoch is:

    tau            the learned threshold matrix (or scalar)
    L_magnitude    (||g*|| - sum_i ||g_i||)^2      -- how far the intervened
                                                      update is from the raw sum
    L_progress     -sum_i alpha_i <g*, g_i>        -- alignment of the combined
                                                      step with each task
    L_policy, L_guide, per-task train loss

Those are the policy's RESPONSE to conflict, not conflict itself. This script
plots them honestly under that name. To plot true conflict evolution you have
to log the cosine matrix during training (about five lines in _step_aim) and
re-run -- see the note printed at the end.

Panels (all: Uni-Mol vs GNN, x = epoch)
    1. mean off-diagonal tau, with the min-max band across task pairs
    2. L_magnitude, indexed to epoch 1 (the two backbones differ by ~100x in
       absolute terms, so the shape is what is comparable)
    3. L_progress,  indexed to epoch 1

Usage
-----
    python plot_conflict_evolution.py                  # 11 tasks, AIM-Matrix
    python plot_conflict_evolution.py --tasks 2        # 2-task runs
    python plot_conflict_evolution.py --aim scalar
    python plot_conflict_evolution.py --smooth 10      # rolling mean over epochs
    python plot_conflict_evolution.py --no-save

Saves into figures/:
    policy_evolution_<variant>_<tasks>task_n<N>_seed<S>.png
    policy_evolution_<variant>_<tasks>task_n<N>_seed<S>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from results_table import discover  # noqa: E402

SERIES = {"Uni-Mol": "#2a78d6", "GNN": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

AIM_PREFIX = {"matrix": "aim_matrix", "scalar": "aim_scalar"}
AIM_LABEL = {"matrix": "AIM-Matrix", "scalar": "AIM-Scalar"}


def _style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 140,
    })


# ---------------------------------------------------------------------------

def _tau_stats(tau, band: str) -> tuple:
    """(mean, lo, hi) over the off-diagonal entries; scalar tau -> all equal.

    With 11 tasks a few pairs run far from the rest, so min-max hides the mean;
    the interquartile band is the readable default.
    """
    a = np.asarray(tau, dtype=float)
    if a.ndim == 0:
        v = float(a)
        return v, v, v
    off = a[~np.eye(a.shape[0], dtype=bool)]
    if band == "minmax":
        return float(off.mean()), float(off.min()), float(off.max())
    return (float(off.mean()),
            float(np.percentile(off, 25)), float(np.percentile(off, 75)))


def load_trace(base: Path, prefix: str, n_train: int, seed: int,
               want_tasks: int, band: str = "iqr") -> Optional[dict]:
    f = base / f"{prefix}_n{n_train}_seed{seed}" / "history.json"
    if not f.exists():
        return None
    hist = json.loads(f.read_text())
    if not hist or "tau" not in hist[0]:
        return None
    if len(hist[0]["test_per_task"]) != want_tasks:
        return None

    ep, tmean, tlo, thi, lmag, lprog = [], [], [], [], [], []
    for e in hist:
        m, lo, hi = _tau_stats(e["tau"], band)
        ep.append(e["epoch"])
        tmean.append(m)
        tlo.append(lo)
        thi.append(hi)
        lmag.append(e.get("L_magnitude", np.nan))
        lprog.append(e.get("L_progress", np.nan))

    return dict(run=f.parent, epoch=np.array(ep), tau=np.array(tmean),
                tau_lo=np.array(tlo), tau_hi=np.array(thi),
                l_mag=np.array(lmag, dtype=float),
                l_prog=np.array(lprog, dtype=float),
                n_tasks=len(hist[0]["test_per_task"]))


def collect(args) -> Dict[str, dict]:
    trees = discover({"GNN": args.gnn, "Uni-Mol": args.unimol},
                     args.n_train, args.seed, args.results_root)
    prefix = AIM_PREFIX[args.aim]
    out: Dict[str, dict] = {}
    for backbone, sets in trees.items():
        for base in sets:
            tr = load_trace(base, prefix, args.n_train, args.seed, args.tasks,
                            args.band)
            if tr is not None:
                out[backbone] = tr
                break
    return out


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, k, mode="valid")


def _index(y: np.ndarray) -> np.ndarray:
    """Index to the first finite value so shapes are comparable across scales."""
    finite = y[np.isfinite(y)]
    if finite.size == 0 or finite[0] == 0:
        return y
    return y / abs(finite[0])


# ---------------------------------------------------------------------------

def draw(fig, axes, traces: Dict[str, dict], args) -> None:
    for b in [x for x in SERIES if x in traces]:   # fixed series order
        tr = traces[b]
        c = SERIES[b]
        ep = tr["epoch"]

        axes[0].plot(ep, _smooth(tr["tau"], args.smooth), color=c, lw=2,
                     zorder=3, label=b)
        if args.band != "none":
            axes[0].fill_between(ep, tr["tau_lo"], tr["tau_hi"], color=c,
                                 alpha=0.12, linewidth=0, zorder=2)

        axes[1].plot(ep, _smooth(_index(tr["l_mag"]), args.smooth),
                     color=c, lw=2, zorder=3, label=b)
        axes[2].plot(ep, _smooth(_index(tr["l_prog"]), args.smooth),
                     color=c, lw=2, zorder=3, label=b)

    axes[0].axhline(0, color=AXIS, lw=1.0, zorder=1)
    axes[0].set_ylabel("mean off-diagonal $\\tau$")
    band_txt = "min-max" if args.band == "minmax" else "inter-quartile"
    axes[0].set_title(f"Learned threshold $\\tau$   (band: {band_txt} across pairs)",
                      color=INK, loc="left", pad=8)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("L_magnitude  (indexed, epoch 1 = 1)")
    axes[1].set_title("Magnitude penalty  $(\\|g^*\\| - \\Sigma_i \\|g_i\\|)^2$",
                      color=INK, loc="left", pad=8)

    axes[2].set_ylabel("L_progress  (indexed, epoch 1 = 1)")
    axes[2].set_title("Progress term  $-\\Sigma_i \\alpha_i \\langle g^*, g_i \\rangle$",
                      color=INK, loc="left", pad=8)

    for ax in axes:
        ax.set_xlabel("epoch")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)

    # Absolute end values, since panels 2 and 3 are indexed.
    notes = []
    for b in [x for x in SERIES if x in traces]:
        tr = traces[b]
        fin = lambda a: a[np.isfinite(a)][-1] if np.isfinite(a).any() else float("nan")
        notes.append(f"{b}: final τ {tr['tau'][-1]:+.3f}, "
                     f"L_mag {fin(tr['l_mag']):.3g}, L_prog {fin(tr['l_prog']):.3g}")
    fig._abs_notes = "   ·   ".join(notes)


def make_figure(traces: Dict[str, dict], args) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    draw(fig, axes, traces, args)

    fig.suptitle("How the AIM policy evolves during training",
                 x=0.010, y=0.975, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.010, 0.915,
             f"{args.tasks} QM9 tasks · {AIM_LABEL[args.aim]} · "
             f"Uni-Mol (pretrained) vs GNN (from scratch) · "
             f"n_train={args.n_train}, seed {args.seed}"
             + (f" · {args.smooth}-epoch rolling mean" if args.smooth > 1 else ""),
             fontsize=9.5, color=MUTED, ha="left")

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right",
               bbox_to_anchor=(0.99, 0.985), ncol=len(names))

    fig.text(0.010, 0.015,
             "These are the policy's response to conflict, not conflict itself: "
             "cos(g_i, g_j) is never logged by train.py.   "
             + getattr(fig, "_abs_notes", ""),
             fontsize=7.5, color=MUTED, ha="left")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.16, wspace=0.26)
    return fig


def save_csv(path: Path, traces: Dict[str, dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "epoch", "tau_mean", "tau_min", "tau_max",
                    "L_magnitude", "L_progress"])
        for b, tr in traces.items():
            for i, e in enumerate(tr["epoch"]):
                w.writerow([b, int(e), f"{tr['tau'][i]:.6f}",
                            f"{tr['tau_lo'][i]:.6f}", f"{tr['tau_hi'][i]:.6f}",
                            f"{tr['l_mag'][i]:.6g}", f"{tr['l_prog'][i]:.6g}"])


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="AIM policy evolution across epochs, Uni-Mol vs GNN.")
    ap.add_argument("--tasks", type=int, default=11, choices=[2, 11],
                    help="which task set to read (default 11)")
    ap.add_argument("--aim", default="matrix", choices=["matrix", "scalar"])
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band", default="iqr", choices=["iqr", "minmax", "none"],
                    help="spread shown around the mean tau (default: iqr)")
    ap.add_argument("--smooth", type=int, default=1,
                    help="rolling-mean window in epochs (1 = raw)")
    ap.add_argument("--gnn", default=None)
    ap.add_argument("--unimol", default=None)
    ap.add_argument("--results_root", default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    traces = collect(args)
    if not traces:
        sys.exit(
            f"No {args.tasks}-task {AIM_PREFIX[args.aim]} runs with a logged tau "
            f"found for n_train={args.n_train}, seed={args.seed}.\n"
            f"Only AIM runs record tau; LS/PCGrad/STL have no policy.")

    for b, tr in traces.items():
        print(f"  {b:9s} {tr['run'].name}  {len(tr['epoch'])} epochs  "
              f"tau {tr['tau'][0]:+.3f} -> {tr['tau'][-1]:+.3f}")

    fig = make_figure(traces, args)

    if args.no_save:
        print("\n--no-save: figure not written")
        return

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stem = (f"policy_evolution_{args.aim}_{args.tasks}task_"
            f"n{args.n_train}_seed{args.seed}")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    save_csv(out / f"{stem}.csv", traces)
    print(f"\nsaved {out / (stem + '.png')}")
    print(f"saved {out / (stem + '.csv')}")
    print("\nFor TRUE conflict evolution you need cos(g_i, g_j) logged per epoch:\n"
          "  in train.py::_step_aim, after flat_grads is built, accumulate the\n"
          "  cosine matrix into the epoch log next to 'tau', then re-run.")


if __name__ == "__main__":
    main()
