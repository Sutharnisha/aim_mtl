"""
Run the AIM-paper method x seed matrix on the 10k QM9 subset (Uni-Mol
backbone) and produce the paper-style comparison table.

Methods run here (by request, a subset of the paper's full baseline list):
  - STL (one run per task), LS, PCGrad, AIM (scalar), AIM (matrix)

Paper protocol (Minot & Schneider, NeurIPS AI4Science 2025; Sec. 4):
  - Subset  : QM9, n_train = 10,000
  - Seeds   : paper uses N = 3 independent runs, mean +/- std + paired
              significance vs STL. Default here is a SINGLE seed (42) —
              faster (~13-36h vs ~40-108h for 3 seeds), but no std/
              significance testing (needs >= 2 seeds; skipped gracefully
              with 1). Pass --seeds 42 43 44 to restore the full protocol.
  - Metrics : Mean Rank (MR), Delta_m% vs STL, per-task test MAE

Usage (from Unimol/src/):
    python run_10k_experiments.py                            # 1 seed (42), defaults
    python run_10k_experiments.py --seeds 42 43 44            # full paper protocol
    python run_10k_experiments.py --n_epochs 5 --patience 5 --seeds 42   # smoke test
    python run_10k_experiments.py --tasks mu eps_LUMO         # 2-task variant (all methods)

With --tasks, every run (STL + MTL) is saved under
    <save_dir>/tasks_<name1>-<name2>-.../
so subset and full 11-task runs never collide.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from train import train
from data import TASK_NAMES, N_TASKS, parse_task_spec
from metrics import aggregate_seeds, paired_significance, print_results_table, best_epoch_key

MTL_METHODS = ["ls", "pcgrad", "aim_scalar", "aim_matrix"]
N_TRAIN = 10_000


def _load_history(save_dir: str, run_name: str) -> Optional[List[dict]]:
    """history.json of a finished run under save_dir, or None if absent."""
    path = Path(save_dir) / run_name / "history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _best_test_per_task(history: List[dict]) -> Dict[str, float]:
    """Test-set MAE per task at the epoch train.py selected (val_select:
    normalized val MAE, mean over tasks for MTL, own task for STL) — i.e.
    the same checkpoint saved as best_model.pt."""
    best = min(history, key=best_epoch_key)
    return best["test_per_task"]


def run_stl(seed: int, save_dir: str, resume: bool = False, **kw) -> Dict[str, float]:
    """One STL run per task; task i's MAE comes only from the run that
    actually trained on task i (see plot_stl_result.py's note on this).
    With resume=True, a task whose run folder already holds a history.json
    is loaded instead of retrained."""
    per_task = {}
    task_cols = kw.get("task_indices") or list(range(N_TASKS))   # kw may carry task_indices
    for i in task_cols:
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


def run_mtl(method: str, seed: int, save_dir: str, resume: bool = False, **kw) -> Dict[str, float]:
    run_name = f"{method}_n{N_TRAIN}_seed{seed}"
    history = _load_history(save_dir, run_name) if resume else None
    if history is not None:
        print(f"\n--- seed {seed} | {method} — loaded from {run_name} ---")
    else:
        print(f"\n--- seed {seed} | {method} ---")
        history = train(method=method, n_train=N_TRAIN, seed=seed, save_dir=save_dir, **kw)
    return _best_test_per_task(history)


def main():
    p = argparse.ArgumentParser(description="AIM paper 10k QM9 method x seed matrix")
    p.add_argument("--save_dir",  default="../results_2task")
    p.add_argument("--n_epochs",  type=int, default=400)
    p.add_argument("--patience",  type=int, default=75)
    p.add_argument("--seeds",     type=int, nargs="+", default=[42])
    p.add_argument("--methods",   nargs="*", default=MTL_METHODS,
                    help="MTL methods to run in addition to per-task STL")
    p.add_argument("--data_root", default="../../data/qm9")
    p.add_argument("--tasks",     nargs="+", default=None,
                    help="Task names/indices to use (default: all 11). E.g. "
                         "--tasks mu eps_LUMO runs the 2-task problem for every "
                         "method; results go to <save_dir>/tasks_mu-eps_LUMO/")
    p.add_argument("--skip_stl",  action="store_true",
                    help="Run only the MTL methods, no STL baseline at all: the "
                         "table then reports per-task MAE and Mean Rank only "
                         "(no Delta_m%% vs STL, no significance test)")
    p.add_argument("--resume",    action="store_true",
                    help="Skip any run (STL or MTL) whose save_dir folder already "
                         "has a history.json and load its results instead")
    args = p.parse_args()

    task_cols  = parse_task_spec(args.tasks)
    task_names = [TASK_NAMES[i] for i in task_cols]
    save_dir   = args.save_dir
    if len(task_cols) != N_TASKS:
        save_dir = str(Path(args.save_dir) / ("tasks_" + "-".join(task_names)))
    print(f"Tasks ({len(task_cols)}): {task_names}  |  save_dir: {save_dir}")

    train_kw = dict(n_epochs=args.n_epochs, patience=args.patience,
                     data_root=args.data_root, task_indices=task_cols)

    seed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed in args.seeds:
        sid = str(seed)
        seed_results[sid] = {}
        if not args.skip_stl:
            seed_results[sid]["stl"] = run_stl(
                seed, save_dir, resume=args.resume, **train_kw)
        for method in args.methods:
            seed_results[sid][method] = run_mtl(
                method, seed, save_dir, resume=args.resume, **train_kw)

    out_path = Path(save_dir) / f"seed_results_n{N_TRAIN}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(seed_results, f, indent=2)
    print(f"\nRaw per-seed results saved: {out_path}")

    if not any(seed_results[s] for s in seed_results):
        print("Nothing was run (--skip_stl with an empty --methods list); no table.")
        return

    # ── Aggregate + report (paper-style table) ────────────────────────────
    agg = aggregate_seeds(seed_results)
    mean_results = {m: {t: agg[m][t]["mean"] for t in agg[m]} for m in agg}
    std_results  = {m: {t: agg[m][t]["std"]  for t in agg[m]} for m in agg}

    print(f"\n{'='*80}\nMean test MAE across {len(args.seeds)} seeds "
          f"(n_train={N_TRAIN})\n{'='*80}")
    print_results_table(mean_results, stl_baseline=mean_results.get("stl"),
                        task_names=task_names)

    print("\nStd across seeds:")
    for method in mean_results:
        std_str = "  ".join(f"{t}={std_results[method][t]:.4f}" for t in task_names)
        print(f"  {method:12s}: {std_str}")

    if args.skip_stl:
        print("\n(--skip_stl: no STL baseline, so no Delta_m% or significance test.)")
    elif len(args.seeds) >= 2:
        print("\nSignificance vs STL (paired t-test across seeds, alpha=0.05):")
        for method in args.methods:
            sig = paired_significance(seed_results, method, "stl")
            n_sig = sum(1 for t in sig.values() if t["significant"])
            print(f"  {method:12s}: significant on {n_sig}/{len(sig)} tasks")
    else:
        print("\n(Need >= 2 seeds for significance testing — only "
              f"{len(args.seeds)} seed given.)")


if __name__ == "__main__":
    main()
