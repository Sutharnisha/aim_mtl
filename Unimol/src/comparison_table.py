"""
comparison_table.py — Mean Rank + Δm% (vs STL / LS / PCGrad) for BOTH backbones.

Reads the 11-task, n_train=10k runs of
  Uni-Mol : ../results/<method>_n<N>_seed<S>/history.json
  GNN     : ../../GNN/result_updated/<method>_n<N>_seed<S>/history.json
and prints one table per backbone, plus a CSV and a PNG.

The STL baseline per task comes from the per-task STL runs
(stl_task<i>_<name>_n<N>_seed<S>); a missing STL run leaves that task's
Δm% as NaN. Best epoch = the checkpoint train.py selected (metrics.best_epoch_key).

Run from Unimol/src/:
    python comparison_table.py                      # n_train=10000, seed=42
    python comparison_table.py --n_train 10000 --seed 43
"""

import argparse
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
from data import TASK_NAMES, TASK_UNITS
from metrics import mean_rank, delta_m_percent, best_epoch_key

# ── Config ────────────────────────────────────────────────────────────────────

BACKBONES = {
    "Uni-Mol": Path("../results"),
    "GNN":     Path("../../GNN/result_updated"),
}
METHODS = {          # display label → run-name prefix
    "LS":         "ls",
    "PCGrad":     "pcgrad",
    "AIM Scalar": "aim_scalar",
    "AIM Matrix": "aim_matrix",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_history(run_dir: Path) -> Optional[list]:
    f = run_dir / "history.json"
    if not f.exists():
        return None
    with open(f) as fh:
        h = json.load(fh)
    return h or None


def load_backbone(base: Path, n_train: int, seed: int):
    """Return (results {label: {task: mae}}, stl {task: mae}) — missing runs skipped."""
    results: Dict[str, Dict[str, float]] = {}
    for label, prefix in METHODS.items():
        h = _load_history(base / f"{prefix}_n{n_train}_seed{seed}")
        if h is None:
            print(f"  [{base}] missing run {prefix}_n{n_train}_seed{seed} — skipped")
            continue
        results[label] = min(h, key=best_epoch_key)["test_per_task"]

    stl: Dict[str, float] = {}
    for i, task in enumerate(TASK_NAMES):
        h = _load_history(base / f"stl_task{i}_{task}_n{n_train}_seed{seed}")
        if h is None:
            stl[task] = float("nan")
            continue
        # STL: own-task val MAE picks the epoch (same rule as train.py's val_select)
        best = min(h, key=lambda e: e.get("val_select", e["val_per_task"][task]))
        stl[task] = best["test_per_task"][task]
    return results, stl


def compute(results, stl):
    """MR over STL + all MTL methods present; Δm% vs STL, LS and PCGrad."""
    all_r = {"STL": stl, **results}
    mr    = mean_rank(all_r)
    dm    = {"STL": delta_m_percent(all_r, stl)}
    for ref in ("LS", "PCGrad"):
        dm[ref] = delta_m_percent(all_r, results[ref]) if ref in results else {m: float("nan") for m in all_r}
    return all_r, mr, dm


def build_rows(all_r, mr, dm):
    rows = []
    for m in ["STL"] + [k for k in METHODS if k in all_r]:
        rows.append([m, mr[m], dm["STL"][m], dm["LS"][m], dm["PCGrad"][m]]
                    + [all_r[m].get(t, float("nan")) for t in TASK_NAMES])
    return rows

# ── Printing / export ─────────────────────────────────────────────────────────

COL_LABELS = ["Method", "MR", "Dm%(STL)", "Dm%(LS)", "Dm%(PCGrad)"] + \
             [f"{t}({TASK_UNITS[t]})" for t in TASK_NAMES]


def fmt(val, col):
    if isinstance(val, str):    return val
    if np.isnan(val):           return "n/a"
    if col == 1:                return f"{val:.2f}"
    if 2 <= col <= 4:           return f"{val:+.1f}%"
    return f"{val:.4f}" if abs(val) < 1 else f"{val:.3f}"


def print_table(title, rows):
    widths = [11, 5, 9, 9, 11] + [10] * len(TASK_NAMES)
    hdr = "  ".join(f"{h:<{w}}" for h, w in zip(COL_LABELS, widths))
    print(f"\n{'=' * len(hdr)}\n  {title}\n{'=' * len(hdr)}\n{hdr}\n{'-' * len(hdr)}")
    for row in rows:
        print("  ".join(f"{fmt(v, i):<{w}}" for i, (v, w) in enumerate(zip(row, widths))))


def make_png_table(ax, rows, title):
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=6)
    n_cols = len(COL_LABELS)
    text = [[fmt(v, i) for i, v in enumerate(row)] for row in rows]
    tbl = ax.table(cellText=text, colLabels=COL_LABELS, loc="center", cellLoc="center",
                   colWidths=[0.09, 0.05, 0.07, 0.07, 0.08] + [0.058] * len(TASK_NAMES))
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.8)
    for j in range(n_cols):
        c = tbl[0, j]; c.set_facecolor("#2c3e50"); c.set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows):
        for j in range(n_cols):
            c = tbl[i + 1, j]
            c.set_facecolor("#fef9e7" if row[0] == "STL" else ("#ffffff" if i % 2 else "#f4f6f7"))
    # bold the best (lowest MR / MAE, highest Δm%) in each numeric column
    for j in range(1, n_cols):
        vals = np.array([row[j] for row in rows], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        best = int(np.nanargmax(vals)) if 2 <= j <= 4 else int(np.nanargmin(vals))
        tbl[best + 1, j].set_text_props(fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out_dir", default="../results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_backbone = {}
    for name, base in BACKBONES.items():
        results, stl = load_backbone(base, args.n_train, args.seed)
        if not results:
            print(f"No MTL runs found for {name} in {base}; skipping.")
            continue
        all_r, mr, dm = compute(results, stl)
        rows = build_rows(all_r, mr, dm)
        per_backbone[name] = rows
        print_table(f"{name} backbone  (n_train={args.n_train}, seed={args.seed}, "
                    f"MR over {len(all_r)} methods incl. STL)", rows)

    if not per_backbone:
        return

    csv_path = out_dir / f"comparison_table_n{args.n_train}_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Backbone"] + COL_LABELS)
        for name, rows in per_backbone.items():
            for row in rows:
                w.writerow([name] + [fmt(v, i) for i, v in enumerate(row)])

    fig, axes = plt.subplots(len(per_backbone), 1, figsize=(22, 3.2 * len(per_backbone)), squeeze=False)
    fig.suptitle(f"Mean Rank and Δm% — 11-task QM9, n_train={args.n_train}, seed={args.seed}",
                 fontsize=12, fontweight="bold")
    for ax, (name, rows) in zip(axes.flat, per_backbone.items()):
        make_png_table(ax, rows, f"Backbone: {name}")
    png_path = out_dir / f"comparison_table_n{args.n_train}_seed{args.seed}.png"
    plt.tight_layout(); fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved CSV : {csv_path}\nSaved PNG : {png_path}")


if __name__ == "__main__":
    main()
