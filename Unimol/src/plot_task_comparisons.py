"""
Create comparison figures for the saved 2-task and 11-task experiments.

Each figure contains:
  1. validation MAE curves for every multi-task method, one panel per task;
  2. best-epoch test MAE, including the matching STL runs when available;
  3. the AIM-Matrix tau heatmap at the AIM-Matrix best epoch.

Run from ``src/``:
    python plot_task_comparisons.py
    python plot_task_comparisons.py --n_train 10000 --seed 42

The output files are written to ``../plots`` by default.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "LS": "ls",
    "PCGrad": "pcgrad",
    "AIM Scalar": "aim_scalar",
    "AIM Matrix": "aim_matrix",
}
COLORS = {
    "LS": "#4878CF",
    "PCGrad": "#6ACC65",
    "AIM Scalar": "#D65F5F",
    "AIM Matrix": "#B47CC7",
    "STL": "#888888",
}


def _load_history(run_dir: Path):
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    with history_path.open() as file:
        history = json.load(file)
    return history or None


def _selection_value(entry: dict) -> float:
    """Match train.py's normalized validation metric when it is available."""
    return float(entry.get("val_select", entry.get("val_mae_norm_mean", entry["val_mae_mean"])))


def _best_entry(history: list) -> dict:
    return min(history, key=_selection_value)


def _find_stl_history(results_dir: Path, task: str, n_train: int, seed: int):
    pattern = f"stl_task*_{task}_n{n_train}_seed{seed}"
    matches = sorted(results_dir.glob(pattern))
    return _load_history(matches[0]) if matches else None


def _task_units(tasks: list[str]) -> dict[str, str]:
    units = {
        "mu": "D", "alpha": "Bohr^3", "eps_HOMO": "eV", "eps_LUMO": "eV",
        "R2": "Bohr^2", "zpve": "eV", "U0": "eV", "U": "eV",
        "H": "eV", "G": "eV", "Cv": "cal/(mol*K)",
    }
    return {task: units.get(task, "") for task in tasks}


def plot_task_comparison(results_dir: Path, output_path: Path, n_train: int, seed: int):
    histories = {}
    for label, prefix in METHODS.items():
        history = _load_history(results_dir / f"{prefix}_n{n_train}_seed{seed}")
        if history is not None:
            histories[label] = history

    if not histories:
        print(f"No multi-task histories found in {results_dir}; skipped.")
        return False

    first_history = next(iter(histories.values()))
    tasks = list(first_history[0]["val_per_task"])
    units = _task_units(tasks)
    stl_histories = {
        task: _find_stl_history(results_dir, task, n_train, seed)
        for task in tasks
    }
    best = {label: _best_entry(history) for label, history in histories.items()}

    n_cols = min(4, len(tasks))
    n_rows = (len(tasks) + n_cols - 1) // n_cols
    figure = plt.figure(figsize=(4.2 * n_cols, 4.0 * (n_rows + 2)))
    grid = figure.add_gridspec(
        n_rows + 2, n_cols, hspace=0.58, wspace=0.35,
        left=0.05, right=0.98, top=0.94, bottom=0.05,
    )
    figure.suptitle(
        f"Uni-Mol task comparison — {len(tasks)} tasks, "
        f"n_train={n_train}, seed={seed}",
        fontsize=14, fontweight="bold",
    )

    for index, task in enumerate(tasks):
        axis = figure.add_subplot(grid[index // n_cols, index % n_cols])
        for label, history in histories.items():
            epochs = [entry["epoch"] for entry in history]
            values = [entry["val_per_task"].get(task, np.nan) for entry in history]
            axis.plot(epochs, values, color=COLORS[label], linewidth=1.3, label=label)
        stl_history = stl_histories[task]
        if stl_history is not None:
            axis.plot(
                [entry["epoch"] for entry in stl_history],
                [entry["val_per_task"].get(task, np.nan) for entry in stl_history],
                ":", color=COLORS["STL"], linewidth=1.4, label="STL",
            )
        unit = f" ({units[task]})" if units[task] else ""
        axis.set_title(f"{task}{unit}", fontsize=10)
        axis.set_xlabel("Epoch", fontsize=8)
        axis.set_ylabel("Validation MAE", fontsize=8)
        axis.grid(True, alpha=0.3)
        if index == 0:
            axis.legend(fontsize=7, ncol=2)

    # Best-epoch test MAE. STL uses its own task-specific validation minimum.
    axis = figure.add_subplot(grid[n_rows, :])
    labels = list(histories) + ["STL"]
    x = np.arange(len(tasks))
    width = 0.8 / len(labels)
    for offset, label in enumerate(labels):
        values = []
        for task in tasks:
            history = stl_histories[task] if label == "STL" else histories[label]
            if history is None:
                values.append(np.nan)
                continue
            entry = min(history, key=lambda item: item["val_per_task"].get(task, np.inf)) if label == "STL" else best[label]
            values.append(entry["test_per_task"].get(task, np.nan))
        axis.bar(x + (offset - (len(labels) - 1) / 2) * width, values,
                 width, label=label, color=COLORS[label], alpha=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels(tasks)
    axis.set_ylabel("Test MAE at selected epoch")
    axis.set_title("Best-epoch test MAE (lower is better)")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(ncol=min(5, len(labels)), fontsize=8)

    # AIM-Matrix policy heatmap, if that run is present.
    axis = figure.add_subplot(grid[n_rows + 1, :])
    aim_entry = best.get("AIM Matrix")
    if aim_entry is None or "tau" not in aim_entry:
        axis.axis("off")
        axis.text(0.5, 0.5, "AIM-Matrix history is unavailable", ha="center", va="center")
    else:
        tau = np.asarray(aim_entry["tau"], dtype=float)
        if tau.ndim == 0:
            tau = np.full((len(tasks), len(tasks)), float(tau))
        masked_tau = np.ma.masked_where(np.eye(len(tasks), dtype=bool), tau)
        image = axis.imshow(masked_tau, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(tasks)), tasks, rotation=45, ha="right")
        axis.set_yticks(range(len(tasks)), tasks)
        for row in range(len(tasks)):
            for column in range(len(tasks)):
                if row != column:
                    axis.text(column, row, f"{tau[row, column]:.2f}",
                              ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="tau_ij", shrink=0.8)
        axis.set_title(f"AIM-Matrix tau at best epoch {aim_entry['epoch']}")
        axis.set_xlabel("Task j")
        axis.set_ylabel("Task i")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")
    return True


def plot_aim_policy_diagnostics(
    results_dir: Path,
    output_path: Path,
    n_train: int,
    seed: int,
    early_epoch: int = 10,
    late_epoch: int = 100,
):
    """Plot the four AIM policy heatmaps in the reference figure layout.

    ``tau`` is already saved by the current training loop. The other three
    matrices must be logged as ``conflict_rate``, ``cosine_similarity`` and
    ``projection_weight`` for this figure to contain all four diagnostics.
    """
    run_dir = results_dir / f"aim_matrix_n{n_train}_seed{seed}"
    history = _load_history(run_dir)
    if history is None:
        print(f"No AIM-Matrix history found in {run_dir}; skipped diagnostics.")
        return False

    task_names = list(history[0]["val_per_task"])
    diagnostic_names = [
        ("Threshold Matrix (tau)", "tau", "RdBu_r", -1, 1),
        ("Conflict Rate (Cos. Sim. < tau)", "conflict_rate", "Blues", 0, 1),
        ("Avg. Cosine Similarity", "cosine_similarity", "RdBu_r", -1, 1),
        ("Avg. Projection Weight (w_proj)", "projection_weight", "YlOrBr", 0, 1),
    ]
    missing = [label for label, key, *_ in diagnostic_names
               if key != "tau" and key not in history[0]]
    if missing:
        print(
            f"AIM diagnostics are not present in {run_dir / 'history.json'}: "
            + ", ".join(missing)
            + ". The training loop currently saves tau only; rerun training "
              "after logging these matrices to render the complete figure."
        )
        return False

    def nearest_entry(target_epoch):
        return min(history, key=lambda entry: abs(entry["epoch"] - target_epoch))

    entries = [nearest_entry(early_epoch), nearest_entry(late_epoch)]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    figure.suptitle(
        f"Evolution of the learned AIM policy — {len(task_names)} tasks\n"
        f"{run_dir.name}", fontsize=13, fontweight="bold",
    )

    for row, entry in enumerate(entries):
        for column, (title, key, cmap, vmin, vmax) in enumerate(diagnostic_names):
            axis = axes[row][column]
            matrix = np.asarray(entry[key], dtype=float)
            if matrix.ndim == 0:
                matrix = np.full((len(task_names), len(task_names)), float(matrix))
            image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axis.set_xticks(range(len(task_names)), task_names, rotation=90, fontsize=6)
            axis.set_yticks(range(len(task_names)), task_names, fontsize=6)
            for task_row in range(len(task_names)):
                for task_column in range(len(task_names)):
                    if task_row != task_column:
                        axis.text(task_column, task_row, f"{matrix[task_row, task_column]:.2f}",
                                  ha="center", va="center", fontsize=5)
            axis.set_title(title, fontsize=9)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axes[row][0].set_ylabel(f"Epoch {entry['epoch']}\nTask i", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_root", default="..")
    parser.add_argument("--out_dir", default="../plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    for task_count in (2, 11):
        results_dir = results_root / f"results_{task_count}task"
        output_path = out_dir / f"comparison_{task_count}task_n{args.n_train}_seed{args.seed}.png"
        plot_task_comparison(results_dir, output_path, args.n_train, args.seed)
        diagnostic_path = out_dir / f"aim_policy_diagnostics_{task_count}task_n{args.n_train}_seed{args.seed}.png"
        plot_aim_policy_diagnostics(results_dir, diagnostic_path, args.n_train, args.seed)


if __name__ == "__main__":
    main()