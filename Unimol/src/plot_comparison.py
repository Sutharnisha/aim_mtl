"""
plot_comparison.py — cross-backbone figure for the 11-task QM9 10k setting.

  Panel A : validation MAE curves, one subplot per task (3x4 grid);
            Uni-Mol methods solid, GNN methods dashed, STL dotted grey.
  Panel B : Δm% vs STL per method, one bar group per backbone.
  Panel C : AIM-Matrix τ heatmap at the best epoch, one per backbone.

Run from Unimol/src/:
    python plot_comparison.py [--n_train 10000 --seed 42]
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import TASK_NAMES, TASK_UNITS, N_TASKS
from metrics import delta_m_percent, best_epoch_key
from comparison_table import BACKBONES, METHODS, load_backbone, _load_history

COLORS = {"LS": "#4878CF", "PCGrad": "#6ACC65", "AIM Scalar": "#D65F5F",
          "AIM Matrix": "#B47CC7", "STL": "#888888"}
STYLE  = {"Uni-Mol": "-", "GNN": "--"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out_dir", default="../results")
    args = ap.parse_args()

    hist, best, stl, stl_hist = {}, {}, {}, {}
    for bb, base in BACKBONES.items():
        hist[bb] = {}
        for label, prefix in METHODS.items():
            h = _load_history(base / f"{prefix}_n{args.n_train}_seed{args.seed}")
            if h is not None:
                hist[bb][label] = h
        best[bb] = {l: min(h, key=best_epoch_key) for l, h in hist[bb].items()}
        _, stl[bb] = load_backbone(base, args.n_train, args.seed)
        stl_hist[bb] = {}
        for i, t in enumerate(TASK_NAMES):
            h = _load_history(base / f"stl_task{i}_{t}_n{args.n_train}_seed{args.seed}")
            if h is not None:
                stl_hist[bb][t] = h
    backbones = [bb for bb in BACKBONES if hist[bb]]
    if not backbones:
        print("No runs found for either backbone."); return

    fig = plt.figure(figsize=(20, 22))
    gs  = fig.add_gridspec(5, 4, hspace=0.55, wspace=0.35, top=0.95, bottom=0.04, left=0.05, right=0.98)
    fig.suptitle(f"AIM vs baselines — 11-task QM9, n_train={args.n_train}, seed={args.seed}  "
                 f"(Uni-Mol solid, GNN dashed)", fontsize=14, fontweight="bold")

    # Panel A: val MAE curves per task
    for i, t in enumerate(TASK_NAMES):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        for bb in backbones:
            for label, h in hist[bb].items():
                ax.plot([e["epoch"] for e in h], [e["val_per_task"][t] for e in h],
                        STYLE[bb], color=COLORS[label], linewidth=1.3,
                        label=f"{label} ({bb})")
            if t in stl_hist[bb]:
                h = stl_hist[bb][t]
                ax.plot([e["epoch"] for e in h], [e["val_per_task"][t] for e in h],
                        ":", color=COLORS["STL"], linewidth=1.3, label=f"STL ({bb})")
        ax.set_title(f"{t} ({TASK_UNITS[t]})", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Val MAE", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6, ncol=2)

    # Panel B: Δm% vs STL, grouped by backbone
    ax = fig.add_subplot(gs[3, :])
    labels = list(METHODS.keys()); x = np.arange(len(labels)); w = 0.8 / len(backbones)
    for k, bb in enumerate(backbones):
        results = {l: b["test_per_task"] for l, b in best[bb].items()}
        dm = delta_m_percent({**results, "STL": stl[bb]}, stl[bb])
        vals = [dm.get(l, np.nan) for l in labels]
        bars = ax.bar(x + (k - (len(backbones) - 1) / 2) * w, vals, w, label=bb,
                      color=[COLORS[l] for l in labels], alpha=0.9 if k == 0 else 0.5,
                      hatch=None if k == 0 else "//", edgecolor="black")
        for b_, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{v:+.1f}%",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Δm% vs STL (positive = better than STL)")
    ax.set_title("Δm% vs per-task STL baselines (NaN tasks skipped)", fontsize=10)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    # Panel C: τ heatmaps
    for k, bb in enumerate(backbones):
        ax = fig.add_subplot(gs[4, 2 * k: 2 * k + 2])
        b = best[bb].get("AIM Matrix")
        if b is None or "tau" not in b:
            ax.axis("off"); ax.set_title(f"{bb}: no AIM-Matrix run", fontsize=10); continue
        tau = np.array(b["tau"])
        if tau.ndim == 0:
            tau = np.full((N_TASKS, N_TASKS), float(tau))
        sns.heatmap(tau, ax=ax, cmap="RdBu_r", center=0.0, vmin=-1, vmax=1,
                    mask=np.eye(N_TASKS, dtype=bool), annot=True, fmt=".2f",
                    annot_kws={"size": 6}, xticklabels=TASK_NAMES, yticklabels=TASK_NAMES,
                    cbar_kws={"label": "τ_ij", "shrink": 0.8}, linewidths=0.3)
        ax.set_title(f"{bb} AIM-Matrix τ (best epoch {b['epoch']})", fontsize=10)
        ax.set_xlabel("Task j"); ax.set_ylabel("Task i")

    out = Path(args.out_dir) / f"comparison_all_n{args.n_train}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
