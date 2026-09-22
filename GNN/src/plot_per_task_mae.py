"""
Per-task test MAE, one bar per method -- MPNN (GNN) backbone.

    "How does each method do on each individual property?"

Reads history.json for every run in --results_dir, takes the epoch train.py
actually checkpointed (metrics.best_epoch_key), and plots the per-task test MAE
in PHYSICAL units. No checkpoints, no torch, no measurement -- the numbers are
already on disk.

Why the form changes with the task count
----------------------------------------
Raw MAE across the 11-task set spans four orders of magnitude (zpve ~0.005 eV,
U0 ~10 eV). Grouped bars on one shared axis would render zpve, mu and the two
orbital energies as invisible slivers next to the energies -- the chart would be
about which property has the biggest numbers, not about the methods. So:

    <= 3 tasks : one panel, grouped bars, shared axis (scales are comparable)
    >  3 tasks : small multiples, one panel per task, each with its OWN y-axis

Both keep physical units, which is what makes the numbers checkable against the
tables. If you want a single shared axis across all eleven, normalise by the STL
baseline first -- that is a different plot (relative), not this one (absolute).

STL is drawn as a reference line rather than a fifth bar: it is the baseline the
other four are judged against, and keeping it out of the bar group holds the
categorical palette to four validated slots.

Usage (from GNN/src/)
---------------------
    python plot_per_task_mae.py --results_dir ../result_11task
    python plot_per_task_mae.py --results_dir ../result_2task
    python plot_per_task_mae.py --results_dir E:/Projects/Results/GNN/result_11task
    python plot_per_task_mae.py --tasks mu eps_LUMO      # subset of what is there
    python plot_per_task_mae.py --out ../gnn_plots

Writes into --out:
    per_task_mae_<N>task_n<n>_seed<s>.png
    per_task_mae_<N>task_n<n>_seed<s>.csv    the plotted numbers
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Deliberately self-contained: importing data.py would pull in torch and the
# collator, and loading torch's OpenMP runtime next to matplotlib's crashes on
# this machine ("libomp.dll ... libiomp5md.dll already initialized"). Nothing
# here needs either, so the three things we do need are inlined instead. They
# must stay in step with data.py / metrics.py.
TASK_NAMES = ["mu", "alpha", "eps_HOMO", "eps_LUMO", "R2", "zpve",
              "U0", "U", "H", "G", "Cv"]
TASK_UNITS = {"mu": "D", "alpha": "Bohr^3", "eps_HOMO": "eV",
              "eps_LUMO": "eV", "R2": "Bohr^2", "zpve": "eV", "U0": "eV",
              "U": "eV", "H": "eV", "G": "eV", "Cv": "cal/(mol*K)"}


def parse_task_spec(tasks) -> List[int]:
    """Task names or indices -> QM9 column indices (mirrors data.py)."""
    if not tasks:
        return list(range(len(TASK_NAMES)))
    out = []
    for tk in tasks:
        tk = str(tk)
        if tk.lstrip("-").isdigit():
            i = int(tk)
        elif tk in TASK_NAMES:
            i = TASK_NAMES.index(tk)
        else:
            raise ValueError(f"unknown task {tk!r}; pick from {TASK_NAMES}")
        if not 0 <= i < len(TASK_NAMES):
            raise ValueError(f"task index {i} out of range")
        out.append(i)
    return out


def best_epoch_key(entry: dict) -> float:
    """The epoch train.py checkpointed (mirrors metrics.best_epoch_key)."""
    return entry.get("val_select",
                     entry.get("val_mae_norm_mean", entry["val_mae_mean"]))

BACKBONE = "MPNN (GNN)"
DEFAULT_RESULTS = "../result_11task"

# Display label -> run prefix. Order is the reading order and the colour order.
METHODS = {
    "LS": "ls",
    "PCGrad": "pcgrad",
    "AIM-Scalar": "aim_scalar",
    "AIM-Matrix": "aim_matrix",
}
# Validated categorical slots 1-4 (adjacent CVD dE 9.1, normal-vision 22.9).
# Two of them sit below 3:1 on the light surface, so every bar carries a visible
# value label -- that is the documented relief for the contrast warning.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
STL_LINE = "#52514e"


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
        "xtick.labelsize": 8.5, "ytick.labelsize": 8,
        "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------

def _best(run_dir: Path) -> Optional[dict]:
    f = run_dir / "history.json"
    if not f.exists():
        return None
    hist = json.loads(f.read_text())
    return min(hist, key=best_epoch_key) if hist else None


def load(results_dir: Path, n_train: int, seed: int) -> dict:
    """{'mtl': {label: {task: mae}}, 'stl': {task: mae}, 'tasks': [...]}"""
    mtl: Dict[str, Dict[str, float]] = {}
    for label, prefix in METHODS.items():
        e = _best(results_dir / f"{prefix}_n{n_train}_seed{seed}")
        if e is not None:
            mtl[label] = dict(e["test_per_task"])

    stl: Dict[str, float] = {}
    for d in sorted(results_dir.glob(f"stl_task*_n{n_train}_seed{seed}")):
        e = _best(d)
        if e is None:
            continue
        m = re.match(rf"stl_task\d+_(.+)_n{n_train}_seed{seed}$", d.name)
        if m and m.group(1) in e["test_per_task"]:
            stl[m.group(1)] = e["test_per_task"][m.group(1)]

    keys = set().union(*(set(v) for v in mtl.values())) if mtl else set(stl)
    tasks = [t for t in TASK_NAMES if t in keys]
    return dict(mtl=mtl, stl=stl, tasks=tasks)


# ---------------------------------------------------------------------------

def _bars(ax, tasks, data, labels, show_units: bool) -> None:
    """One grouped-bar cluster per task on a shared axis (few tasks only)."""
    x = np.arange(len(tasks))
    w = 0.8 / max(len(labels), 1)
    for s, label in enumerate(labels):
        vals = [data["mtl"][label].get(t, np.nan) for t in tasks]
        off = (s - (len(labels) - 1) / 2) * w
        ax.bar(x + off, vals, w * 0.9, color=COLOURS[s], edgecolor=SURFACE,
               linewidth=1.2, zorder=3, label=label)
        for xi, v in zip(x + off, vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, color=SECONDARY, rotation=90)
    for i, t in enumerate(tasks):
        if t in data["stl"]:
            ax.plot([i - 0.45, i + 0.45], [data["stl"][t]] * 2,
                    color=STL_LINE, lw=1.4, zorder=5)
    ax.set_xticks(x, [f"{t}\n({TASK_UNITS.get(t, '')})" if show_units else t
                      for t in tasks])
    ax.set_ylabel("test MAE (physical units)")
    ax.margins(y=0.22)


def _panel(ax, task, data, labels) -> None:
    """One task, its own y-axis -- the only honest way across 11 scales."""
    vals = [data["mtl"][l].get(task, np.nan) for l in labels]
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.68, color=COLOURS[:len(labels)], edgecolor=SURFACE,
           linewidth=1.2, zorder=3)
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7,
                    color=SECONDARY)
    if task in data["stl"]:
        ax.axhline(data["stl"][task], color=STL_LINE, lw=1.3, zorder=5)
        ax.text(len(labels) - 0.5, data["stl"][task], " STL", va="center",
                ha="left", fontsize=7, color=STL_LINE)
    ax.set_xticks([])
    ax.set_title(f"{task}   ({TASK_UNITS.get(task, '')})", color=INK,
                 loc="left", pad=6)
    ax.margins(y=0.24)


def make_figure(data: dict, args) -> plt.Figure:
    _style()
    labels = [l for l in METHODS if l in data["mtl"]]
    tasks = data["tasks"]
    n = len(tasks)

    if n <= 3:
        fig, ax = plt.subplots(figsize=(3.2 + 2.1 * n, 4.6))
        _bars(ax, tasks, data, labels, show_units=True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", ncol=len(labels))
        axes_top = 0.80
    else:
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, t in zip(axes, tasks):
            _panel(ax, t, data, labels)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(True, axis="y", zorder=0)
            ax.set_axisbelow(True)
        for ax in axes[n:]:
            ax.set_axis_off()
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[i])
                   for i in range(len(labels))]
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.99, 0.985), ncol=len(labels))
        axes_top = 0.88

    fig.suptitle(f"Per-task test MAE  \u2014  {BACKBONE}",
                 x=0.010, y=0.985, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.010, 0.945 if n > 3 else 0.925,
             f"{n} QM9 tasks \u00b7 n_train={args.n_train}, seed {args.seed} \u00b7 "
             f"lower is better \u00b7 horizontal rule = STL baseline"
             + ("  \u00b7  each panel has its own y-axis: the tasks span four "
                "orders of magnitude" if n > 3 else ""),
             fontsize=9, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=axes_top, bottom=0.10,
                        hspace=0.52, wspace=0.28)
    return fig


def save_csv(path: Path, data: dict) -> None:
    labels = [l for l in METHODS if l in data["mtl"]]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Method"] + data["tasks"])
        if data["stl"]:
            w.writerow(["STL"] + [f"{data['stl'].get(t, float('nan')):.4f}"
                                  for t in data["tasks"]])
        for l in labels:
            w.writerow([l] + [f"{data['mtl'][l].get(t, float('nan')):.4f}"
                              for t in data["tasks"]])


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Per-task test MAE bar plot, {BACKBONE}.")
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS,
                    help="run tree, e.g. ../result_11task or ../result_2task")
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="restrict to a subset of the tasks found")
    ap.add_argument("--out", default="../plots")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    if not rd.is_dir():
        sys.exit(f"{rd} is not a directory -- pass --results_dir")

    data = load(rd, args.n_train, args.seed)
    if not data["mtl"]:
        sys.exit(f"no {args.n_train}/{args.seed} runs found in {rd}")

    if args.tasks:
        want = [TASK_NAMES[i] for i in parse_task_spec(args.tasks)]
        data["tasks"] = [t for t in data["tasks"] if t in want]

    print(f"{BACKBONE}  <-  {rd}")
    print(f"  methods : {', '.join(data['mtl'])}")
    print(f"  tasks   : {len(data['tasks'])}  ({', '.join(data['tasks'])})")
    print(f"  STL     : {len(data['stl'])} task(s) available as a baseline")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"per_task_mae_{len(data['tasks'])}task_n{args.n_train}_seed{args.seed}"

    fig = make_figure(data, args)
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    save_csv(out / f"{stem}.csv", data)
    print(f"\nsaved {out / (stem + '.png')}")
    print(f"saved {out / (stem + '.csv')}")


if __name__ == "__main__":
    main()
