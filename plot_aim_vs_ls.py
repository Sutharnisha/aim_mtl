"""
Per-task AIM improvement over LS -- which individual properties actually benefit?

For every QM9 task t:

    improvement_t = 100 * (MAE_LS,t - MAE_AIM,t) / MAE_LS,t

Positive means AIM beats linear scalarization on that property; negative means
LS was better. One bar per task per backbone, so the question the figure answers
is not "is AIM better on average" (that is the Mean Rank / delta-m% table) but
"WHERE does it help, and is that the same place on a pretrained encoder as on
one trained from scratch".

LS is the reference rather than STL on purpose: LS is the honest do-nothing
multi-task baseline -- same architecture, same budget, no gradient surgery --
so the difference isolates the intervention itself.

Run loading is imported from results_table.py, so both tools agree on which
epoch each run is read at (metrics.best_epoch_key -- the epoch train.py
actually checkpointed).

Usage
-----
    python plot_aim_vs_ls.py                       # 11-task, AIM-Matrix vs LS
    python plot_aim_vs_ls.py --aim scalar
    python plot_aim_vs_ls.py --aim both            # both variants, small multiples
    python plot_aim_vs_ls.py --tasks 2
    python plot_aim_vs_ls.py --order task          # keep the canonical task order
    python plot_aim_vs_ls.py --labels all          # a number on every bar
    python plot_aim_vs_ls.py --no-save             # show/print only

Saves by default into figures/:
    aim_vs_ls_<variant>_<tasks>task_n<N>_seed<S>.png
    aim_vs_ls_<variant>_<tasks>task_n<N>_seed<S>.csv   the plotted numbers
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from results_table import discover, load_run_set  # noqa: E402

# Validated categorical pair (CVD dE 24.7, normal-vision dE 33.6 on #fcfcfb).
SERIES = {"Uni-Mol": "#2a78d6", "GNN": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

AIM_LABEL = {"matrix": "AIM-Matrix", "scalar": "AIM-Scalar"}
AIM_RUN = {"matrix": "AIM-Matrix", "scalar": "AIM-Scalar"}


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
        "xtick.labelsize": 9, "ytick.labelsize": 9.5,
        "axes.titlesize": 11, "axes.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 140,
    })


# ---------------------------------------------------------------------------

def collect(n_train: int, seed: int, want_tasks: int, aim_label: str,
            overrides: Dict[str, Optional[str]],
            results_root: Optional[str]) -> Optional[dict]:
    """{backbone: {task: improvement %}} plus the task list, or None."""
    trees = discover(overrides, n_train, seed, results_root)

    per_backbone: Dict[str, Dict[str, float]] = {}
    tasks: List[str] = []
    sources: Dict[str, Path] = {}

    for backbone, sets in trees.items():
        for base in sets:
            rs = load_run_set(base, n_train, seed)
            if rs is None or len(rs["tasks"]) != want_tasks:
                continue
            if "LS" not in rs["mtl"] or aim_label not in rs["mtl"]:
                print(f"  !! {backbone}: {base} lacks LS or {aim_label} -- skipped")
                continue
            ls, aim = rs["mtl"]["LS"], rs["mtl"][aim_label]
            per_backbone[backbone] = {
                t: 100.0 * (ls[t] - aim[t]) / ls[t]
                for t in rs["tasks"] if t in ls and t in aim and ls[t] != 0
            }
            sources[backbone] = base
            tasks = tasks or rs["tasks"]
            break

    if not per_backbone:
        return None
    return dict(data=per_backbone, tasks=tasks, sources=sources)


# ---------------------------------------------------------------------------

def draw(ax, bundle: dict, order: str, labels: str, title: str) -> None:
    data, tasks = bundle["data"], bundle["tasks"]
    backbones = [b for b in SERIES if b in data]

    if order == "gain":
        tasks = sorted(
            tasks,
            key=lambda t: np.mean([data[b].get(t, np.nan) for b in backbones]))

    y = np.arange(len(tasks))
    h = 0.38 if len(backbones) > 1 else 0.55

    extremes = set()
    if labels == "auto":
        flat = [(abs(data[b].get(t, 0.0)), b, t) for b in backbones for t in tasks]
        extremes = {(b, t) for _, b, t in sorted(flat, reverse=True)[:4]}

    for s, b in enumerate(backbones):
        vals = [data[b].get(t, np.nan) for t in tasks]
        off = (s - (len(backbones) - 1) / 2) * h
        ax.barh(y + off, vals, h * 0.92, color=SERIES[b],
                edgecolor=SURFACE, linewidth=1.5, zorder=3, label=b)
        for yi, t, v in zip(y + off, tasks, vals):
            if np.isnan(v):
                continue
            if labels == "all" or (b, t) in extremes:
                ax.text(v + (0.6 if v >= 0 else -0.6), yi, f"{v:+.1f}%",
                        va="center", ha="left" if v >= 0 else "right",
                        fontsize=8, color=SECONDARY)

    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=4)
    ax.set_yticks(y, tasks)
    ax.invert_yaxis()
    ax.set_xlabel("AIM better than LS  (%)   →")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.16)
    ax.set_title(title, color=INK, loc="left", pad=10)


def make_figure(bundles: Dict[str, dict], args) -> plt.Figure:
    _style()
    n = len(bundles)
    n_tasks = len(next(iter(bundles.values()))["tasks"])
    height = max(3.2, 0.42 * n_tasks + 2.3)      # 2 tasks -> short, 11 -> tall
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, height), squeeze=False)
    axes = axes[0]

    for ax, (variant, bundle) in zip(axes, bundles.items()):
        draw(ax, bundle, args.order, args.labels,
             f"{AIM_LABEL[variant]} vs LS")

    # Header offsets in inches, so they hold at any figure height.
    y_title = 1 - 0.30 / height
    y_sub = 1 - 0.60 / height
    y_foot = 0.17 / height

    first = next(iter(bundles.values()))
    fig.suptitle("Per-task AIM improvement over linear scalarization",
                 x=0.012, y=y_title, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.012, y_sub,
             f"{len(first['tasks'])} QM9 tasks · Uni-Mol (pretrained) vs GNN "
             f"(from scratch) · n_train={args.n_train}, seed {args.seed}. "
             "Bars right of zero: AIM beats LS on that property.",
             fontsize=9.5, color=MUTED, ha="left")

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right",
               bbox_to_anchor=(0.99, y_title + 0.012), ncol=len(names))

    src = "   ·   ".join(f"{b}: {p}" for b, p in first["sources"].items())
    fig.text(0.012, y_foot,
             f"Single seed — per-task differences of a few % are inside seed "
             f"noise.   {src}", fontsize=7.5, color=MUTED, ha="left")

    fig.subplots_adjust(left=0.10, right=0.97,
                        top=1 - 0.98 / height, bottom=0.62 / height, wspace=0.28)
    return fig


def save_csv(path: Path, bundle: dict) -> None:
    backbones = [b for b in SERIES if b in bundle["data"]]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task"] + [f"{b} AIM-vs-LS %" for b in backbones])
        for t in bundle["tasks"]:
            w.writerow([t] + [f"{bundle['data'][b].get(t, float('nan')):.2f}"
                              for b in backbones])


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-task AIM-vs-LS improvement, Uni-Mol vs GNN.")
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", type=int, default=11, choices=[2, 11])
    ap.add_argument("--aim", default="matrix",
                    choices=["matrix", "scalar", "both"])
    ap.add_argument("--order", default="gain", choices=["gain", "task"],
                    help="sort tasks by mean improvement, or keep task order")
    ap.add_argument("--labels", default="auto", choices=["auto", "all", "none"],
                    help="value labels: the extremes, every bar, or none")
    ap.add_argument("--gnn", default=None)
    ap.add_argument("--unimol", default=None)
    ap.add_argument("--results_root", default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    variants = ["matrix", "scalar"] if args.aim == "both" else [args.aim]
    overrides = {"GNN": args.gnn, "Uni-Mol": args.unimol}

    bundles: Dict[str, dict] = {}
    for v in variants:
        b = collect(args.n_train, args.seed, args.tasks, AIM_RUN[v],
                    overrides, args.results_root)
        if b is None:
            print(f"  !! no {args.tasks}-task runs with LS + {AIM_RUN[v]} found")
            continue
        bundles[v] = b

    if not bundles:
        sys.exit(
            f"Nothing to plot for n_train={args.n_train}, seed={args.seed}, "
            f"{args.tasks} tasks.\nPoint at the runs with --results_root "
            f"../Results  (or --gnn / --unimol).")

    for v, bundle in bundles.items():
        print(f"\n{AIM_LABEL[v]} vs LS  ({args.tasks} tasks)")
        for b, d in bundle["data"].items():
            wins = [t for t, x in d.items() if x > 0]
            print(f"  {b:9s} AIM better on {len(wins)}/{len(d)} tasks: "
                  f"{', '.join(wins) if wins else 'none'}")

    fig = make_figure(bundles, args)

    if args.no_save:
        print("\n--no-save: figure not written")
        return

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = args.aim
    stem = f"aim_vs_ls_{tag}_{args.tasks}task_n{args.n_train}_seed{args.seed}"
    png = out / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"\nsaved {png}")

    for v, bundle in bundles.items():
        c = out / (f"{stem}.csv" if len(bundles) == 1
                   else f"aim_vs_ls_{v}_{args.tasks}task_"
                        f"n{args.n_train}_seed{args.seed}.csv")
        save_csv(c, bundle)
        print(f"saved {c}")


if __name__ == "__main__":
    main()
