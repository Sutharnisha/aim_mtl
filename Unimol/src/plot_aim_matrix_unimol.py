"""
AIM-Matrix (Uni-Mol only) — training diagnostics for one run.

Produces <results_dir>/<run_name>/aim_matrix_unimol.png with:
  Rows 1-3 : validation MAE curves, one subplot per task (11 tasks, 3x4 grid)
  Row 4    : normalized val MAE (the model-selection metric, best epoch marked)
             + AIM-Matrix τ heatmap (11x11) at the best epoch

Run from Unimol/src/:
    python plot_aim_matrix_unimol.py                                  # aim_matrix_n10000_seed42
    python plot_aim_matrix_unimol.py --run_name aim_matrix_n10000_seed43
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import TASK_NAMES, TASK_UNITS, N_TASKS
from metrics import best_epoch_key

COLOR = "#B47CC7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--run_name",    default="aim_matrix_n10000_seed42")
    args = ap.parse_args()

    run_dir = Path(args.results_dir) / args.run_name
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    best   = min(history, key=best_epoch_key)
    epochs = [e["epoch"] for e in history]

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f"AIM-Matrix (Uni-Mol) — {args.run_name}, {len(history)} epochs",
                 fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.35, left=0.05, right=0.98, top=0.94, bottom=0.05)

    for i, task in enumerate(TASK_NAMES):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.plot(epochs, [e["val_per_task"][task] for e in history], color=COLOR, linewidth=1.4)
        ax.axvline(best["epoch"], color=COLOR, alpha=0.3, linestyle=":")
        ax.set_title(f"Val MAE — {task} ({TASK_UNITS[task]})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("MAE", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 0:2])
    sel = [best_epoch_key(e) for e in history]
    ax.plot(epochs, sel, color=COLOR, linewidth=1.6)
    ax.scatter([best["epoch"]], [best_epoch_key(best)], color="black", zorder=5, s=30,
               label=f"best: epoch {best['epoch']} ({best_epoch_key(best):.4f})")
    ax.set_title("Normalized val MAE (model-selection metric)", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("MAE / std", fontsize=8)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 2:4])
    tau = np.array(best["tau"])
    if tau.ndim == 0:
        tau = np.full((N_TASKS, N_TASKS), float(tau))
    sns.heatmap(tau, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 6},
                cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.3,
                mask=np.eye(N_TASKS, dtype=bool),
                xticklabels=TASK_NAMES, yticklabels=TASK_NAMES,
                cbar_kws={"label": "τ_ij (conflict threshold)", "shrink": 0.8})
    ax.set_title(f"AIM-Matrix τ (best epoch = {best['epoch']})", fontsize=10)
    ax.set_xlabel("Task j"); ax.set_ylabel("Task i")

    out = run_dir / "aim_matrix_unimol.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
