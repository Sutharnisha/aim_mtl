"""
Run ONLY the single-task-learning (STL) baseline on the 10k QM9 subset
(GNN backbone): one independent run per task, for each seed.

This is the STL half of run_10k_experiments.py, split out so the STL
baseline can be trained on its own (e.g. on a separate machine / job)
without touching the MTL methods. Run folders and names are identical to
those produced by run_10k_experiments.py, so a later
    python run_10k_experiments.py --resume
will pick these runs up instead of retraining them.

Usage (from GNN/src/):
    python run_10k_stl.py                              # all 11 tasks, seed 42
    python run_10k_stl.py --seeds 42 43 44             # full paper protocol
    python run_10k_stl.py --tasks 0 3 6                # only a subset of tasks
    python run_10k_stl.py --tasks mu U0 --resume       # by name; skip finished runs
    python run_10k_stl.py --n_epochs 5 --patience 5    # smoke test
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from train import train
from data import TASK_NAMES, N_TASKS
from metrics import aggregate_seeds
from run_10k_experiments import N_TRAIN, _load_history, _best_test_per_task


def _parse_tasks(tasks: List[str]) -> List[int]:
    """Accept task indices ('3') or task names ('U0'); return sorted indices."""
    idxs = []
    for t in tasks:
        if t.isdigit():
            i = int(t)
            if not 0 <= i < N_TASKS:
                raise ValueError(f"task index {i} out of range 0..{N_TASKS - 1}")
        elif t in TASK_NAMES:
            i = TASK_NAMES.index(t)
        else:
            raise ValueError(f"unknown task '{t}'; choose from {TASK_NAMES}")
        idxs.append(i)
    return sorted(set(idxs))


def run_stl_tasks(task_idxs: List[int], seed: int, save_dir: str,
                  resume: bool = False, **kw) -> Dict[str, float]:
    """One STL run per requested task; task i's MAE comes only from the run
    that trained on task i. With resume=True, a task whose run folder already
    holds a history.json is loaded instead of retrained."""
    per_task = {}
    for i in task_idxs:
        run_name = f"stl_task{i}_{TASK_NAMES[i]}_n{N_TRAIN}_seed{seed}"
        history = _load_history(save_dir, run_name) if resume else None
        if history is not None:
            print(f"\n--- seed {seed} | stl task {i} ({TASK_NAMES[i]}) — loaded from {run_name} ---")
        else:
            print(f"\n--- seed {seed} | stl task {i} ({TASK_NAMES[i]}) ---")
            history = train(
                method="stl", stl_task_idx=i, n_train=N_TRAIN,
                seed=seed, save_dir=save_dir, **kw,
            )
        per_task[TASK_NAMES[i]] = _best_test_per_task(history)[TASK_NAMES[i]]
    return per_task


def main():
    p = argparse.ArgumentParser(description="STL-only baseline on 10k QM9 (GNN)")
    p.add_argument("--save_dir",  default="../result_stl")
    p.add_argument("--n_epochs",  type=int, default=400)
    p.add_argument("--patience",  type=int, default=75)
    p.add_argument("--seeds",     type=int, nargs="+", default=[42])
    p.add_argument("--tasks",     nargs="+", default=None,
                    help="Task indices (0-10) or names to run; default: all tasks")
    p.add_argument("--data_root", default="../../data/qm9")
    p.add_argument("--resume",    action="store_true",
                    help="Skip any task whose save_dir run folder already has a "
                         "history.json and load its result instead")
    args = p.parse_args()

    task_idxs = list(range(N_TASKS)) if args.tasks is None else _parse_tasks(args.tasks)
    task_names = [TASK_NAMES[i] for i in task_idxs]
    train_kw = dict(n_epochs=args.n_epochs, patience=args.patience,
                     data_root=args.data_root)

    print(f"STL-only run | n_train={N_TRAIN} | seeds={args.seeds} | "
          f"tasks={task_names}")

    # Same nesting as run_10k_experiments.py ({seed: {method: {task: mae}}})
    # so metrics.aggregate_seeds / downstream scripts can consume it directly.
    seed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed in args.seeds:
        seed_results[str(seed)] = {
            "stl": run_stl_tasks(task_idxs, seed, args.save_dir,
                                 resume=args.resume, **train_kw)
        }

    out_path = Path(args.save_dir) / f"stl_results_n{N_TRAIN}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(seed_results, f, indent=2)
    print(f"\nRaw per-seed STL results saved: {out_path}")

    # ── Per-task summary table ────────────────────────────────────────────
    agg = aggregate_seeds(seed_results)["stl"]
    print(f"\n{'='*60}\nSTL test MAE across {len(args.seeds)} seed(s) "
          f"(n_train={N_TRAIN})\n{'='*60}")
    print(f"{'task':>10s}  {'mean MAE':>12s}  {'std':>10s}")
    for t in task_names:
        print(f"{t:>10s}  {agg[t]['mean']:12.6f}  {agg[t]['std']:10.6f}")
    if len(args.seeds) < 2:
        print("\n(std is 0 / undefined with a single seed; pass --seeds 42 43 44 "
              "for the full protocol.)")


if __name__ == "__main__":
    main()
