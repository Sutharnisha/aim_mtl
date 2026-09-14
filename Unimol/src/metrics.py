"""
Evaluation metrics from the AIM paper.

Mean Rank (MR):
    For each task, rank all methods by MAE (rank 1 = best).
    MR for a method = average rank across all tasks.
    Lower MR is better.

Delta_m% (Δm%):
    Mean percentage improvement of method over the STL baseline.
    Δm% = (1/T) * sum_i  [ sign_i * (stl_i - method_i) / stl_i * 100 ]
    where sign_i = +1 when lower is better (MAE regression), -1 otherwise.
    Higher Δm% is better (more positive = bigger improvement).

Seed aggregation + significance (paper Sec. 4: "N = 3 independent runs
with different random seeds", reporting mean +/- std and flagging
statistically significant improvements over STL):
    aggregate_seeds()   — mean/std of MAE per (method, task) across seeds.
    paired_significance() — paired t-test of a method's per-seed MAE
                             against STL's per-seed MAE, per task.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional


TaskMAE = Dict[str, float]          # {task_name: mae_value}
Results = Dict[str, TaskMAE]        # {method_name: TaskMAE}
SeedResults = Dict[str, Results]    # {seed_id: {method: {task: mae}}}


def best_epoch_key(entry: dict) -> float:
    """
    Sort key selecting a run's best epoch from its history.json — the same
    rule train.py uses for best_model.pt: 'val_select' (normalized val MAE,
    mean over tasks for MTL runs, the trained task only for STL runs).
    Falls back for histories written by older code.
    """
    return entry.get("val_select",
           entry.get("val_mae_norm_mean",
           entry["val_mae_mean"]))


def mean_rank(results: Results) -> Dict[str, float]:
    """
    Compute Mean Rank for every method.

    Args:
        results : {method: {task: mae}}

    Returns:
        {method: mean_rank}  — lower is better
    """
    methods = list(results.keys())
    tasks   = list(next(iter(results.values())).keys())

    rank_lists: Dict[str, List[int]] = {m: [] for m in methods}

    for task in tasks:
        # Sort methods by MAE on this task (ascending = lower MAE is better)
        ordered = sorted(methods, key=lambda m: results[m].get(task, float("inf")))
        for rank, method in enumerate(ordered, start=1):
            rank_lists[method].append(rank)

    return {m: float(np.mean(rank_lists[m])) for m in methods}


def delta_m_percent(
    results:        Results,
    stl_baseline:   TaskMAE,
    higher_is_better: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Δm% for every method relative to single-task learning (STL).

    Δm%_method = (1/T) * sum_i  (stl_i - method_i) / |stl_i| * 100

    Interpretation: positive Δm% means the method is BETTER than STL.
    This implementation uses the more intuitive convention where higher Δm% = better.

    Args:
        results          : {method: {task: mae}}
        stl_baseline     : {task: mae}  from single-task training
        higher_is_better : list of task names where higher metric is better
                           (none for MAE-only benchmarks)

    Returns:
        {method: delta_m_percent}
    """
    if higher_is_better is None:
        higher_is_better = []

    methods = list(results.keys())
    tasks   = list(stl_baseline.keys())

    delta = {}
    for method in methods:
        pct_improvements = []
        for task in tasks:
            stl_val    = stl_baseline[task]
            method_val = results[method].get(task, float("nan"))
            if abs(stl_val) < 1e-12:
                continue
            if task in higher_is_better:
                pct = (method_val - stl_val) / abs(stl_val) * 100.0
            else:
                # For MAE: improvement = stl > method
                pct = (stl_val - method_val) / abs(stl_val) * 100.0
            pct_improvements.append(pct)

        delta[method] = float(np.mean(pct_improvements)) if pct_improvements else float("nan")

    return delta


def aggregate_seeds(
    seed_results: SeedResults,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate per-seed results into mean/std per (method, task).

    Args:
        seed_results : {seed_id: {method: {task: mae}}}

    Returns:
        {method: {task: {"mean": ..., "std": ..., "n": n_seeds}}}
    """
    seeds   = list(seed_results.keys())
    methods = list(seed_results[seeds[0]].keys())
    tasks   = list(seed_results[seeds[0]][methods[0]].keys())

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in methods:
        out[method] = {}
        for task in tasks:
            vals = np.array([seed_results[s][method][task] for s in seeds])
            out[method][task] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n":    len(vals),
            }
    return out


def paired_significance(
    seed_results:  SeedResults,
    method:        str,
    stl_method:    str = "stl",
    alpha:         float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Paired t-test of `method` vs `stl_method`, per task, across seeds
    (paper Sec. 4: "statistically significant improvements over ... STL").
    Requires >= 2 seeds; with N=3 (paper default) this is a low-power test
    reported for consistency with the paper's protocol, not as strong
    statistical evidence on its own.

    Args:
        seed_results : {seed_id: {method: {task: mae}}}
        method       : method name to test
        stl_method   : baseline method name (default "stl")
        alpha        : significance threshold

    Returns:
        {task: {"t": t_stat, "p": p_value, "significant": bool}}
    """
    seeds = list(seed_results.keys())
    if len(seeds) < 2:
        raise ValueError("paired_significance needs >= 2 seeds")
    tasks = list(seed_results[seeds[0]][method].keys())

    out = {}
    for task in tasks:
        a = np.array([seed_results[s][method][task]     for s in seeds])
        b = np.array([seed_results[s][stl_method][task]  for s in seeds])
        if np.allclose(a, b):
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat, p_val = stats.ttest_rel(a, b)
        out[task] = {
            "t": float(t_stat),
            "p": float(p_val),
            "significant": bool(p_val < alpha),
        }
    return out


def print_results_table(
    results:      Results,
    stl_baseline: Optional[TaskMAE] = None,
    task_names:   Optional[List[str]] = None,
):
    """Pretty-print the full results table."""
    methods = list(results.keys())
    if task_names is None:
        task_names = list(next(iter(results.values())).keys())

    mr = mean_rank(results)
    dm = delta_m_percent(results, stl_baseline) if stl_baseline else {}
    has_delta = stl_baseline is not None

    col_w = 9
    header_parts = [f"{'Method':<22}", f"{'MR':>5}", f"{'Δm%':>7}"]
    for t in task_names:
        header_parts.append(f"{t:>{col_w}}")
    print("  ".join(header_parts))
    print("-" * (22 + 5 + 7 + (col_w + 2) * len(task_names) + 10))

    for method in sorted(methods, key=lambda m: mr.get(m, 999)):
        delta_text = f"{dm.get(method, 0):>7.2f}%" if has_delta else f"{'N/A':>7}"
        row = [f"{method:<22}", f"{mr.get(method, 0):>5.1f}", delta_text]
        for t in task_names:
            val = results[method].get(t, float("nan"))
            row.append(f"{val:>{col_w}.4f}")
        print("  ".join(row))
