"""
Build the full output tree: every table and figure, filed by task set and backbone.

    outputs/
      2task/
        tables/        overall comparison + per-task MAE (md + csv)
        comparison/    figures that show both backbones together
        UniMol/        Uni-Mol-only matrices (csv)
        GNN/           GNN-only matrices (csv)
      11task/
        (same)

Why comparison/ exists
----------------------
Most of these figures ARE the comparison -- per-task AIM-vs-LS bars, the
conflict-evolution curves, the side-by-side heatmaps with their difference
panel. Splitting them into per-backbone folders would destroy the thing they
are for. So single-backbone artefacts (the raw matrices) go in GNN/ and
UniMol/, and anything that puts the two side by side lives in comparison/.

What goes where
---------------
  tables/      results_table.py          overall performance comparison
  comparison/  plot_aim_vs_ls.py         per-task AIM improvement over LS
               plot_conflict_evolution.py  policy evolution across epochs
               plot_cosine_matrix.py     cosine similarity, conflict frequency,
                                         AIM intervention weight w_ij
  <backbone>/  the mean matrices behind those heatmaps, as CSV

Two-interpreter reality
-----------------------
The heatmaps need a MEASUREMENT pass (one forward + N backward per batch on a
checkpoint) that only runs in the env carrying unimol_tools, where matplotlib
crashes. So this script only PLOTS from caches that already exist; it prints the
measure commands for anything missing instead of failing.

    # once per (task set, checkpoint), in the aim_gnn env:
    <aim_gnn python> plot_cosine_matrix.py --measure-only [--tasks mu eps_LUMO]
    <aim_gnn python> plot_cosine_matrix.py --measure-only --method aim_matrix [...]

    # then, in the env with a working matplotlib:
    python make_outputs.py

Usage
-----
    python make_outputs.py                 # both task sets
    python make_outputs.py --tasks 2       # just the 2-task tree
    python make_outputs.py --out outputs   # root of the tree
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
PY = sys.executable

TASK_SETS = {
    2: ["mu", "eps_LUMO"],
    11: None,                      # None = all eleven
}


def run(script: str, extra: List[str], label: str) -> bool:
    cmd = [PY, str(ROOT / script)] + extra
    print(f"\n--- {label}")
    print("    " + " ".join(cmd[1:]))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        if any(k in line for k in ("saved", "wrote", "no cache", "!!", "Error",
                                   "Nothing", "no ")):
            print("    " + line.strip())
    if p.returncode != 0:
        print(f"    (exit {p.returncode})")
    return p.returncode == 0


def split_per_backbone(tables: Path, base: Path, tag: str) -> None:
    """Give GNN/ and UniMol/ their own slice of the per-task MAE table."""
    import csv as _csv
    src = next(tables.glob(f"per_task_{n_dash(tag)}_*.csv"), None)
    if src is None:
        return
    with open(src) as fh:
        rows = list(_csv.reader(fh))
    header, body = rows[0], rows[1:]
    for folder, key in (("UniMol", "Uni-Mol"), ("GNN", "GNN")):
        mine = [r for r in body if r and r[0] == key]
        if not mine:
            continue
        d = base / folder
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"per_task_mae_{tag}.csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(header[1:])            # drop the backbone column
            w.writerows(r[1:] for r in mine)
        print(f"    saved {d / f'per_task_mae_{tag}.csv'}")


def n_dash(tag: str) -> str:
    return tag.replace("task", "-task")


def gradient_summary(base: Path, tag: str, args) -> None:
    """Per-backbone cosine / conflict / w_ij summary from the measurement caches."""
    import csv as _csv
    sys.path.insert(0, str(ROOT))
    figs = ROOT / "figures"
    rows_by_backbone: dict = {}

    if tag == "2task":
        from plot_cosine_2task import load_csv as load2
        import numpy as np
        for method in ("ls", "aim_matrix"):
            f = figs / (f"cosine_2task_{method}_primary_"
                        f"n{args.n_train}_seed{args.seed}.csv")
            if not f.exists():
                continue
            for r in load2(f):
                v = r["cos"]
                row = {"checkpoint": method,
                       "mean_cosine": f"{v.mean():+.4f}",
                       "sd_cosine": f"{v.std():.4f}",
                       "conflict_rate": f"{(v < 0).mean():.3f}"}
                tau = r.get("tau")
                if tau is not None:
                    w = 1 / (1 + np.exp(-10.0 * (tau[None] - v[:, None, None])))
                    m = w.mean(axis=0)
                    row["w_0_from_1"] = f"{m[0, 1]:.3f}"
                    row["w_1_from_0"] = f"{m[1, 0]:.3f}"
                rows_by_backbone.setdefault(r["name"], []).append(row)

    for name, rows in rows_by_backbone.items():
        d = base / name.replace("-", "")
        d.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(d / f"gradient_summary_{tag}.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["checkpoint"] +
                                [k for k in keys if k != "checkpoint"])
            w.writeheader()
            w.writerows(rows)
        print(f"    saved {d / f'gradient_summary_{tag}.csv'}")


def build(n_tasks: int, out_root: Path, args) -> None:
    tasks = TASK_SETS[n_tasks]
    tag = f"{n_tasks}task"
    base = out_root / tag
    tables = base / "tables"
    comp = base / "comparison"
    for d in (tables, comp, base / "UniMol", base / "GNN"):
        d.mkdir(parents=True, exist_ok=True)

    task_args = (["--tasks"] + tasks) if tasks else []
    common = ["--n_train", str(args.n_train), "--seed", str(args.seed)]

    print("\n" + "=" * 74)
    print(f"  {tag}   ->  {base}")
    print("=" * 74)

    # 1. Overall performance table -------------------------------------------
    run("results_table.py",
        ["--tasks", str(n_tasks), "--out", str(tables),
         "--md", str(tables / f"overall_{tag}.md")] + common,
        "overall performance comparison (table)")

    # 2. Per-task AIM improvement over LS ------------------------------------
    run("plot_aim_vs_ls.py",
        ["--tasks", str(n_tasks), "--out", str(comp)] + common,
        "per-task AIM improvement over LS (bar)")

    # 3. Conflict / policy evolution during training -------------------------
    run("plot_conflict_evolution.py",
        ["--tasks", str(n_tasks), "--out", str(comp), "--smooth", "5"] + common,
        "policy evolution during training (line)")

    # 4-6. The three heatmaps, from whichever caches exist --------------------
    #      LS checkpoint  -> cosine similarity + conflict frequency
    #      AIM checkpoint -> the same two, plus w_ij (needs tau)
    #      At 2 tasks the matrix is 1x1 off-diagonal, so the strip plots below
    #      carry the same information with the batch spread visible instead.
    if n_tasks > 2:
        for method, what in (("ls", "cosine similarity + conflict frequency"),
                             ("aim_matrix", "AIM intervention weight w_ij")):
            ok = run("plot_cosine_matrix.py",
                     ["--method", method, "--plot-only", "--out", str(base)]
                     + task_args + common,
                     f"{what}  [{method} checkpoint]")
            if not ok:
                print(f"    MEASURE FIRST:  <aim_gnn python> "
                      f"plot_cosine_matrix.py --measure-only --method {method} "
                      + " ".join(task_args + common))

    # The 2-task cosine view is a strip plot, not a heatmap: one task pair is
    # a single number, and the spread across batches is the informative part.
    if n_tasks == 2:
        run("plot_cosine_2task.py",
            ["--plot-only", "--out", str(comp)] + common,
            "cosine spread across batches (2-task strip)")
        run("plot_aim_weight_2task.py",
            ["--plot-only", "--out", str(comp)] + common,
            "AIM intervention weight (2-task)")

    # 7. Per-backbone artefacts -------------------------------------------
    print("\n--- per-backbone files")
    split_per_backbone(tables, base, tag)
    gradient_summary(base, tag, args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the full output tree.")
    ap.add_argument("--tasks", type=int, nargs="+", default=[2, 11],
                    choices=[2, 11])
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--clean", action="store_true",
                    help="delete the output tree first")
    args = ap.parse_args()

    out_root = ROOT / args.out
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
        print(f"removed {out_root}")

    for n in args.tasks:
        build(n, out_root, args)

    print("\n" + "=" * 74)
    print(f"  tree at {out_root}")
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(out_root)}")


if __name__ == "__main__":
    main()
