"""
Plot STL (single-task learning) training runs.

Each STL run trains the shared encoder + ONE task head only, so the other
two heads in that run's history.json were never trained — their MAE columns
are meaningless noise and are intentionally left out of every plot here.
Each run therefore only ever contributes a plot for the one task it trained.

By default this script auto-discovers every stl_task*_... run present in
results_dir (e.g. stl_task0_mu_n10000_seed42, stl_task1_alpha_n10000_seed42,
... one per task in the 11-task QM9 set, see data.py) and plots them
together, one row per task. Pass --run_name to plot just one run instead.

Run from GNN/src/:
    python plot_stl_result.py                                       # all STL runs found
    python plot_stl_result.py --run_name stl_task0_mu_n10000_seed42  # just one
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import TASK_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trained_task_index(run_name: str, history: list) -> int:
    """
    Figure out which task STL was trained on.

    Prefer parsing it from the run name (stl_task<i>_...); fall back to
    finding the task whose train_task_loss is never exactly zero.
    """
    m = re.match(r"stl_task(\d+)_", run_name)
    if m:
        return int(m.group(1))

    for i in range(len(history[0]["train_task_loss"])):
        if any(entry["train_task_loss"][i] != 0.0 for entry in history):
            return i
    raise ValueError("Could not determine the trained task from history.json")


def discover_stl_runs(results_dir: str) -> List[str]:
    """Find every stl_task*_... run directory in results_dir, sorted by task index."""
    root = Path(results_dir)
    if not root.exists():
        return []
    runs = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("stl_task")]

    def _task_idx(name: str) -> int:
        m = re.match(r"stl_task(\d+)_", name)
        return int(m.group(1)) if m else 999

    return sorted(runs, key=_task_idx)


def _load_run(results_dir: str, run_name: str) -> Tuple[list, int, str]:
    history_file = Path(results_dir) / run_name / "history.json"
    with open(history_file) as f:
        history = json.load(f)
    task_idx  = _trained_task_index(run_name, history)
    task_name = TASK_NAMES[task_idx]
    return history, task_idx, task_name


def _draw_task_panels(ax_loss, ax_mae, history: list, task_idx: int, task_name: str, run_name: str):
    """Draw the train-loss and val/test-MAE panels for one STL run."""
    epochs     = [e["epoch"] for e in history]
    train_loss = [e["train_task_loss"][task_idx] for e in history]
    val_mae    = [e["val_per_task"][task_name] for e in history]
    test_mae   = [e["test_per_task"][task_name] for e in history]
    best       = min(history, key=lambda e: e["val_per_task"][task_name])

    ax_loss.plot(epochs, train_loss, color="tab:blue", linewidth=1.6)
    ax_loss.axvline(best["epoch"], color="tab:blue", alpha=0.25, linestyle=":")
    ax_loss.set_title(f"STL train loss — {task_name}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("L1 loss (normalised)")
    ax_loss.grid(True, alpha=0.3)

    ax_mae.plot(epochs, val_mae, label="Validation", color="tab:orange", linewidth=1.8)
    ax_mae.plot(epochs, test_mae, label="Test", color="tab:green", linewidth=1.8)
    ax_mae.scatter(
        [best["epoch"]], [best["val_per_task"][task_name]],
        color="black", zorder=5, s=30,
        label=f"best epoch {best['epoch']}",
    )
    ax_mae.axvline(best["epoch"], color="black", alpha=0.2, linestyle=":")
    ax_mae.set_title(f"STL MAE — {task_name} (physical units)")
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(fontsize=8)

    print(f"[{run_name}] best epoch {best['epoch']}: "
          f"val_MAE({task_name})={best['val_per_task'][task_name]:.4f}  "
          f"test_MAE({task_name})={best['test_per_task'][task_name]:.4f}")


# ---------------------------------------------------------------------------
# Single-run plot
# ---------------------------------------------------------------------------

def plot_stl_result(
    results_dir: str = "../result_updated",
    run_name:    str = "stl_task0_mu_n5000_seed42",
    save_path:   Optional[str] = None,
):
    """Plot one STL run's trained task (train loss + val/test MAE)."""
    history, task_idx, task_name = _load_run(results_dir, run_name)

    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_task_panels(ax_loss, ax_mae, history, task_idx, task_name, run_name)
    fig.suptitle(
        f"Single-task learning (STL) — task: {task_name}  |  run: {run_name}\n"
        f"(other task heads exist in history.json but were never trained — omitted)",
        fontsize=11,
    )
    plt.tight_layout()

    save_path = save_path or str(Path(results_dir) / run_name / f"stl_{task_name}_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Multi-run plot (mu, U0, U side by side)
# ---------------------------------------------------------------------------

def plot_all_stl_results(
    results_dir: str = "../result_updated",
    run_names:   Optional[List[str]] = None,
    save_path:   Optional[str] = None,
):
    """
    Plot every STL run found in results_dir, one row per task
    (auto-discovers stl_task0_mu_..., stl_task1_U0_..., stl_task2_U_...).
    """
    run_names = run_names or discover_stl_runs(results_dir)
    if not run_names:
        print(f"No stl_task*_... runs found in {results_dir}. "
              f"Train them first (one per task, 0..10), e.g.:\n"
              f"  python train.py --method stl --stl_task_idx 0 --n_train 10000 --n_epochs 400\n"
              f"  python train.py --method stl --stl_task_idx 1 --n_train 10000 --n_epochs 400\n"
              f"  ... (or just run run_10k_experiments.py, which does all 11)")
        return None

    fig, axes = plt.subplots(len(run_names), 2, figsize=(11, 4.5 * len(run_names)), squeeze=False)

    for row, run_name in enumerate(run_names):
        history, task_idx, task_name = _load_run(results_dir, run_name)
        _draw_task_panels(axes[row][0], axes[row][1], history, task_idx, task_name, run_name)

    fig.suptitle(
        "Single-task learning (STL) — all trained tasks\n"
        "(each run only trained one head; untrained heads are omitted)",
        fontsize=12,
    )
    plt.tight_layout()

    save_path = save_path or str(Path(results_dir) / "stl_all_tasks.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot STL training run(s)")
    parser.add_argument("--results_dir", default="../result_updated")
    parser.add_argument("--run_name",    default=None,
                         help="Plot just this one run. Omit to auto-discover and plot all STL runs.")
    parser.add_argument("--save_path",   default=None)
    args = parser.parse_args()

    if args.run_name:
        plot_stl_result(args.results_dir, args.run_name, args.save_path)
    else:
        plot_all_stl_results(args.results_dir, save_path=args.save_path)
