"""
Overall performance comparison: STL / LS / PCGrad / AIM-Scalar / AIM-Matrix
crossed with the two backbones (Uni-Mol pretrained, GNN from scratch).

The question it answers
-----------------------
    Does AIM actually improve prediction performance, and does that answer
    change when the shared encoder is pretrained?

It prints three things per task set (2-task and 11-task):

  1. OVERALL TABLE   one row per method, one column block per backbone:
                     Mean Rank among the four MTL methods, Mean Rank including
                     STL, and delta-m% vs STL (positive = better than STL).
  2. PER-TASK TABLE  test MAE in physical units for every method x task.
  3. VERDICT         best AIM variant vs best non-AIM MTL baseline, per
                     backbone, plus whether any MTL method beat STL at all.

All numbers come from each run's history.json at the epoch train.py actually
checkpointed (metrics.best_epoch_key), so they match best_model.pt. Mean Rank
and delta-m% come from the repo's own metrics.py -- the same definitions
analysis.py and comparison_table.py use.

Relation to the existing tools
------------------------------
Unimol/src/comparison_table.py prints one table PER BACKBONE for the 11-task
set only. This script is the cross-backbone view the write-up needs: one
combined method x backbone matrix, both task sets, STL included in the
ranking, and an explicit verdict line.

Usage
-----
    python results_table.py --tasks 2             # 2-task table, printed and saved
    python results_table.py                       # every task set found
    python results_table.py --seed 43
    python results_table.py --out my_tables       # save somewhere else
    python results_table.py --no-save             # print only
    python results_table.py --gnn GNN/result_updated --unimol Unimol/results

Saves by default into tables/:
    results_<tasks>task_n<N>_seed<S>.md     the whole report
    overall_<tasks>-task_n<N>_seed<S>.csv   the method x backbone grid
    per_task_<tasks>-task_n<N>_seed<S>.csv  test MAE per method x task
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent

# metrics.py is byte-identical in both trees; use the GNN copy.
sys.path.insert(0, str(ROOT / "GNN" / "src"))
from metrics import best_epoch_key, mean_rank, delta_m_percent  # noqa: E402

# Display label -> run-directory prefix. Order is the reading order of the table.
METHODS: Dict[str, str] = {
    "LS": "ls",
    "PCGrad": "pcgrad",
    "AIM-Scalar": "aim_scalar",
    "AIM-Matrix": "aim_matrix",
}
AIM_METHODS = ("AIM-Scalar", "AIM-Matrix")
BASELINE_METHODS = ("LS", "PCGrad")

# The trees have been renamed a few times, and the runs currently live in a
# sibling ../Results/ tree rather than inside the repo; try them all.
CANDIDATES: Dict[str, List[str]] = {
    "GNN": ["GNN/result_updated", "GNN/result_11task", "GNN/result_2task",
            "GNN/results",
            "../Results/GNN/result_updated", "../Results/GNN/result_11task",
            "../Results/GNN/result_2task"],
    "Uni-Mol": ["Unimol/results", "Unimol/results_11task",
                "Unimol/results_2task",
                "../Results/Unimol/results", "../Results/Unimol/results_11task",
                "../Results/Unimol/results_2task"],
}
# Sub-paths tried under an explicit --results_root.
ROOT_LAYOUT: Dict[str, List[str]] = {
    "GNN": ["GNN/result_updated", "GNN/result_11task", "GNN/result_2task"],
    "Uni-Mol": ["Unimol/results", "Unimol/results_11task",
                "Unimol/results_2task"],
}

TASK_ORDER = ["mu", "alpha", "eps_HOMO", "eps_LUMO", "R2", "zpve",
              "U0", "U", "H", "G", "Cv"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _run_sets(base: Path, n_train: int, seed: int) -> List[Path]:
    """A run set is any directory holding <method>_n<N>_seed<S>/history.json."""
    if not base.is_dir():
        return []
    found = []
    for cand in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
        if any((cand / f"{p}_n{n_train}_seed{seed}" / "history.json").exists()
               for p in METHODS.values()):
            found.append(cand)
    return found


def discover(overrides: Dict[str, Optional[str]], n_train: int, seed: int,
             results_root: Optional[str] = None) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for backbone, cands in CANDIDATES.items():
        if overrides.get(backbone):
            roots = [ROOT / overrides[backbone]]
        elif results_root:
            rr = ROOT / results_root
            roots = [rr / sub for sub in ROOT_LAYOUT[backbone]]
        else:
            roots = [ROOT / c for c in cands]
        sets: List[Path] = []
        for r in roots:
            for s in _run_sets(r, n_train, seed):
                if s not in sets:
                    sets.append(s)
        out[backbone] = sets
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _best_entry(run_dir: Path) -> Optional[dict]:
    f = run_dir / "history.json"
    if not f.exists():
        return None
    hist = json.loads(f.read_text())
    return min(hist, key=best_epoch_key) if hist else None


def load_run_set(base: Path, n_train: int, seed: int) -> Optional[dict]:
    """Return {tasks, mtl {label: {task: mae}}, stl {task: mae}, epochs {...}}."""
    mtl: Dict[str, Dict[str, float]] = {}
    epochs: Dict[str, Tuple[int, int]] = {}

    for label, prefix in METHODS.items():
        d = base / f"{prefix}_n{n_train}_seed{seed}"
        e = _best_entry(d)
        if e is None:
            continue
        mtl[label] = dict(e["test_per_task"])
        epochs[label] = (e["epoch"], len(json.loads((d / "history.json").read_text())))

    if not mtl:
        return None

    # The task set is whatever the MTL runs were actually trained on.
    keys = set().union(*(set(v) for v in mtl.values()))
    tasks = [t for t in TASK_ORDER if t in keys] + sorted(keys - set(TASK_ORDER))

    # STL: each stl_task<i>_<name>_... run contributes only its own task.
    stl: Dict[str, float] = {}
    for d in sorted(base.glob(f"stl_task*_n{n_train}_seed{seed}")):
        e = _best_entry(d)
        if e is None:
            continue
        name = d.name[len("stl_task"):].split("_", 1)[1]
        name = name[: -len(f"_n{n_train}_seed{seed}")]
        if name in e["test_per_task"]:
            stl[name] = e["test_per_task"][name]
            epochs[f"STL[{name}]"] = (e["epoch"],
                                      len(json.loads((d / "history.json").read_text())))

    return dict(base=base, tasks=tasks, mtl=mtl, stl=stl, epochs=epochs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarise(rs: dict) -> dict:
    """Mean Rank (MTL-only and incl. STL) and delta-m% for one run set."""
    tasks, mtl, stl = rs["tasks"], rs["mtl"], rs["stl"]
    sub = {m: {t: v[t] for t in tasks if t in v} for m, v in mtl.items()}

    mr_mtl = mean_rank(sub)

    complete_stl = all(t in stl for t in tasks)
    if complete_stl:
        mr_all = mean_rank({**sub, "STL": {t: stl[t] for t in tasks}})
        dm = delta_m_percent(sub, {t: stl[t] for t in tasks})
    else:
        mr_all, dm = {}, {}

    return dict(mr_mtl=mr_mtl, mr_all=mr_all, dm=dm,
                complete_stl=complete_stl, n_stl=len(stl))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], nd: int = 2, star: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}" + ("*" if star else "")


def overall_block(label: str, per_backbone: Dict[str, dict]) -> List[List[str]]:
    """Rows for the combined method x backbone table."""
    backbones = list(per_backbone)
    header = ["Method"]
    for b in backbones:
        header += [f"{b} dm%", f"{b} MR", f"{b} MR+STL"]
    rows = [header]

    best = {}
    for b in backbones:
        s = per_backbone[b]["summary"]
        best[b] = (min(s["mr_mtl"], key=s["mr_mtl"].get) if s["mr_mtl"] else None)

    rows.append(["STL"] + sum(
        [["ref", "--",
          _fmt(per_backbone[b]["summary"]["mr_all"].get("STL"))]
         for b in backbones], []))

    for m in METHODS:
        row = [m]
        for b in backbones:
            s = per_backbone[b]["summary"]
            row += [_fmt(s["dm"].get(m), 1),
                    _fmt(s["mr_mtl"].get(m), 2, star=(best[b] == m)),
                    _fmt(s["mr_all"].get(m))]
        rows.append(row)
    return rows


def per_task_block(label: str, per_backbone: Dict[str, dict],
                   tasks: List[str]) -> List[List[str]]:
    rows = [["Backbone", "Method"] + tasks]
    for b, d in per_backbone.items():
        rs = d["runs"]
        if rs["stl"]:
            rows.append([b, "STL"] +
                        [_fmt(rs["stl"].get(t), 4) for t in tasks])
        for m in METHODS:
            if m in rs["mtl"]:
                rows.append([b, m] +
                            [_fmt(rs["mtl"][m].get(t), 4) for t in tasks])
    return rows


def verdict_lines(per_backbone: Dict[str, dict]) -> List[str]:
    out = []
    for b, d in per_backbone.items():
        s = d["summary"]
        mr = s["mr_mtl"]
        if not mr:
            continue
        aim = min((m for m in AIM_METHODS if m in mr), key=mr.get, default=None)
        base = min((m for m in BASELINE_METHODS if m in mr), key=mr.get, default=None)
        if aim is None or base is None:
            continue
        leads = mr[aim] < mr[base]
        out.append(
            f"  {b:9s} best AIM = {aim} (MR {mr[aim]:.2f})   "
            f"best non-AIM = {base} (MR {mr[base]:.2f})   "
            f"-> AIM {'LEADS' if leads else 'does NOT lead'}")
        if s["dm"]:
            beat = [m for m, v in s["dm"].items() if v > 0]
            out.append(f"            beats STL (dm% > 0): "
                       f"{', '.join(beat) if beat else 'none'}")
        else:
            out.append(f"            dm% unavailable "
                       f"(STL runs found for {s['n_stl']} task(s))")
    return out


SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
HAIRLINE, BAND, ACCENT = "#e1e0d9", "#f6f5f2", "#c8741c"


def save_png(path: Path, rows: List[List[str]], title: str, subtitle: str,
             note: str = "") -> None:
    """Render a table as an image. Hairlines only -- no heavy grid, no zebra."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.dpi": 200,
    })

    n_r, n_c = len(rows), len(rows[0])
    # Column widths from the widest cell, in character units.
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(n_c)]
    total = sum(widths)
    fig_w = max(6.0, 0.105 * total + 0.6)
    row_h = 0.30
    head_h = 1.02 + (0.22 if note else 0.0)
    fig_h = head_h + row_h * n_r + 0.28

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total); ax.set_ylim(0, fig_h)

    x_left, xs = 0.6, []
    for w in widths:
        xs.append(x_left)
        x_left += w
    top = fig_h - head_h

    fig.text(0.6 / total, (fig_h - 0.34) / fig_h, title, ha="left", va="center",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.6 / total, (fig_h - 0.66) / fig_h, subtitle, ha="left",
             va="center", fontsize=8.5, color=MUTED)

    for ri, row in enumerate(rows):
        y = top - row_h * (ri + 0.5)
        if ri == 0:                                   # header band
            ax.add_patch(plt.Rectangle((0, y - row_h / 2), total, row_h,
                                       facecolor=BAND, edgecolor="none"))
        for ci, cell in enumerate(row):
            star = cell.endswith("*")
            txt = cell.rstrip("*")
            ax.text(xs[ci] + (0 if ci == 0 else widths[ci] - 2), y, txt,
                    ha="left" if ci == 0 else "right", va="center",
                    fontsize=9, color=ACCENT if star else INK,
                    fontweight="bold" if (ri == 0 or star) else "normal")
        ax.plot([0, total], [y - row_h / 2] * 2, color=HAIRLINE, lw=0.8,
                zorder=0)

    ax.plot([0, total], [top] * 2, color=SECONDARY, lw=1.0)
    if note:
        ax.text(0.6, 0.22, note, ha="left", va="center", fontsize=7.5,
                color=MUTED)

    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def print_table(rows: List[List[str]], title: str) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print(f"\n{title}")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for i, r in enumerate(rows):
        print("   ".join(c.rjust(w) if j else c.ljust(w)
                         for j, (c, w) in enumerate(zip(r, widths))))
        if i == 0:
            print("-" * (sum(widths) + 3 * (len(widths) - 1)))


def md_table(rows: List[List[str]]) -> str:
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "|" + "|".join("---" for _ in rows[0]) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return f"{head}\n{sep}\n{body}\n"


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overall STL/LS/PCGrad/AIM x Uni-Mol/GNN performance table.")
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gnn", default=None, help="explicit GNN result tree")
    ap.add_argument("--unimol", default=None, help="explicit Uni-Mol result tree")
    ap.add_argument("--results_root", default=None,
                    help="root holding GNN/ and Unimol/ (e.g. ../Results)")
    ap.add_argument("--tasks", default="all", choices=["2", "11", "all"],
                    help="which task set to report (default: all found)")
    ap.add_argument("--out", default="tables",
                    help="directory the tables are saved into (default: tables/)")
    ap.add_argument("--md", default=None,
                    help="write the Markdown to this exact path instead")
    ap.add_argument("--no-save", action="store_true",
                    help="print only, write nothing")
    args = ap.parse_args()

    trees = discover({"GNN": args.gnn, "Uni-Mol": args.unimol},
                     args.n_train, args.seed, args.results_root)

    # Group run sets by task signature so 2-task and 11-task get their own block.
    blocks: Dict[str, Dict[str, dict]] = {}
    for backbone, sets in trees.items():
        for base in sets:
            rs = load_run_set(base, args.n_train, args.seed)
            if rs is None:
                continue
            if args.tasks != "all" and len(rs["tasks"]) != int(args.tasks):
                continue
            key = f"{len(rs['tasks'])}-task ({', '.join(rs['tasks'])})"
            blocks.setdefault(key, {})[backbone] = dict(
                runs=rs, summary=summarise(rs))

    if not blocks:
        looked = "\n".join(
            f"    {b}: " + ", ".join(str(ROOT / c) for c in CANDIDATES[b])
            for b in CANDIDATES)
        sys.exit(
            f"No runs found for n_train={args.n_train}, seed={args.seed}.\n"
            f"Looked in:\n{looked}\n"
            f"Point at the right trees with --gnn / --unimol, e.g.\n"
            f"    python results_table.py --gnn GNN/result_updated "
            f"--unimol Unimol/results")

    md_parts: List[str] = ["# Overall performance comparison\n"]

    for key in sorted(blocks, key=lambda k: int(k.split("-")[0])):
        per_backbone = blocks[key]
        tasks = next(iter(per_backbone.values()))["runs"]["tasks"]

        print("\n" + "=" * 78)
        print(f"  {key}   n_train={args.n_train}  seed={args.seed}")
        print("=" * 78)
        for b, d in per_backbone.items():
            print(f"  {b:9s} <- {d['runs']['base']}")

        overall = overall_block(key, per_backbone)
        print_table(overall,
                    "Overall  (dm%: + = better than STL | MR: lower = better, "
                    "* = best MTL)")

        pertask = per_task_block(key, per_backbone, tasks)
        print_table(pertask, "Per-task test MAE (physical units)")

        print("\nDoes AIM improve prediction performance?")
        for line in verdict_lines(per_backbone):
            print(line)

        md_parts += [f"\n## {key}  (n_train={args.n_train}, seed={args.seed})\n",
                     "\n### Overall\n", md_table(overall),
                     "\n### Per-task test MAE\n", md_table(pertask),
                     "\n### Verdict\n",
                     "\n".join("- " + l.strip() for l in verdict_lines(per_backbone)),
                     "\n"]

        if not args.no_save:
            out = ROOT / args.out
            out.mkdir(parents=True, exist_ok=True)
            tag = key.split(" ")[0]
            titles = {
                "overall": ("Overall performance: STL / LS / PCGrad / AIM "
                            "× Uni-Mol / GNN",
                            "Δm% vs STL (positive = better than "
                            "single-task)   ·   MR = Mean Rank among the "
                            "four MTL methods, lower is better   ·   "
                            "* = best MTL method for that backbone"),
                "per_task": ("Per-task test MAE (physical units)",
                             "At the epoch train.py checkpointed "
                             "(metrics.best_epoch_key)"),
            }
            sub_tail = (f"{key}  ·  n_train={args.n_train}, "
                        f"seed {args.seed}")
            for name, rows in (("overall", overall), ("per_task", pertask)):
                stem = f"{name}_{tag}_n{args.n_train}_seed{args.seed}"
                p = out / f"{stem}.csv"
                with open(p, "w", newline="") as fh:
                    csv.writer(fh).writerows(rows)
                print(f"\nsaved {p}")

                head, sub = titles[name]
                png = out / f"{stem}.png"
                save_png(png, rows, head, f"{sub_tail}   ·   {sub}",
                         note="Single seed — mean-rank gaps below ~0.5 are "
                              "inside seed noise.")
                print(f"saved {png}")

    print("\nNote: single seed. Mean-rank gaps below ~0.5 are inside seed noise; "
          "\n      re-run with --seed 43/44 and compare before claiming a winner.")

    if not args.no_save:
        if args.md:
            p = ROOT / args.md
        else:
            tag = args.tasks if args.tasks != "all" else "all"
            p = (ROOT / args.out /
                 f"results_{tag}task_n{args.n_train}_seed{args.seed}.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(md_parts), encoding="utf-8")
        print(f"saved {p}")


if __name__ == "__main__":
    main()
