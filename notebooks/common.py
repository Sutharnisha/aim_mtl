"""
Shared loading, palette and style for the analysis notebooks.

Deliberately free of torch / rdkit / unimol_tools: everything here reads
history.json or a measurement cache, and loading torch's OpenMP runtime beside
matplotlib's crashes on this machine. Only 10_measure_gradients.ipynb imports
torch, and that one has to run in the conda env that carries unimol_tools.

Edit RESULTS below if the run trees move.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Where things live -- EDIT HERE
# ---------------------------------------------------------------------------

NB_DIR = Path(__file__).resolve().parent
REPO = NB_DIR.parent

RESULTS: Dict[str, Dict[str, Path]] = {
    "Uni-Mol": {
        "2task":  Path(r"E:/Projects/Results/Unimol/results_2task"),
        "11task": Path(r"E:/Projects/Results/Unimol/results_11task"),
    },
    "GNN": {
        "2task":  Path(r"E:/Projects/Results/GNN/result_2task"),
        "11task": Path(r"E:/Projects/Results/GNN/result_11task"),
    },
}

FIG_DIR = NB_DIR / "figures"        # where notebooks write their output
CACHE_DIR = REPO / "figures"        # gradient-measurement caches

N_TRAIN, SEED = 10_000, 42

# ---------------------------------------------------------------------------
# Task metadata (mirrors data.py -- keep in step)
# ---------------------------------------------------------------------------

TASK_NAMES = ["mu", "alpha", "eps_HOMO", "eps_LUMO", "R2", "zpve",
              "U0", "U", "H", "G", "Cv"]
TASK_UNITS = {"mu": "D", "alpha": "Bohr^3", "eps_HOMO": "eV", "eps_LUMO": "eV",
              "R2": "Bohr^2", "zpve": "eV", "U0": "eV", "U": "eV", "H": "eV",
              "G": "eV", "Cv": "cal/(mol*K)"}

METHODS = {"LS": "ls", "PCGrad": "pcgrad",
           "AIM-Scalar": "aim_scalar", "AIM-Matrix": "aim_matrix"}

# ---------------------------------------------------------------------------
# Palette -- validated categorical slots; do not reorder casually
# ---------------------------------------------------------------------------

BACKBONE_COLOUR = {"Uni-Mol": "#2a78d6", "GNN": "#eb6834"}
METHOD_COLOUR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND, ACCENT = "#e1e0d9", "#c3c2b7", "#f6f5f2", "#c8741c"
# Diverging: red (conflict) <-> neutral gray <-> blue (aligned)
DIVERGING = ["#96201f", "#d03b3b", "#f3a09f", "#f0efec",
             "#9ec5f4", "#2a78d6", "#0d366b"]
# Sequential single hue for 0..1 magnitudes
SEQUENTIAL = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec",
              "#3987e5", "#256abf", "#104281"]


def apply_style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 130,
    })


def cmap(kind: str = "diverging"):
    from matplotlib.colors import LinearSegmentedColormap
    cols = DIVERGING if kind == "diverging" else SEQUENTIAL
    c = LinearSegmentedColormap.from_list(kind, cols)
    c.set_bad(SURFACE)
    return c


def save(fig, name: str, subdir: str = "") -> Path:
    d = FIG_DIR / subdir if subdir else FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name if name.endswith(".png") else name + ".png")
    fig.savefig(p, bbox_inches="tight")
    print("saved", p)
    return p


# ---------------------------------------------------------------------------
# Loading runs
# ---------------------------------------------------------------------------

def best_epoch_key(entry: dict) -> float:
    """The epoch train.py checkpointed (mirrors metrics.best_epoch_key)."""
    return entry.get("val_select",
                     entry.get("val_mae_norm_mean", entry["val_mae_mean"]))


def history(results_dir: Path, run_name: str) -> Optional[list]:
    f = Path(results_dir) / run_name / "history.json"
    return json.loads(f.read_text()) if f.exists() else None


def best(results_dir: Path, run_name: str) -> Optional[dict]:
    h = history(results_dir, run_name)
    return min(h, key=best_epoch_key) if h else None


def load_runs(backbone: str, tag: str,
              n_train: int = N_TRAIN, seed: int = SEED) -> dict:
    """{'mtl': {label: {task: mae}}, 'stl': {task: mae}, 'tasks': [...]}"""
    rd = RESULTS[backbone][tag]
    mtl: Dict[str, Dict[str, float]] = {}
    for label, prefix in METHODS.items():
        e = best(rd, f"{prefix}_n{n_train}_seed{seed}")
        if e is not None:
            mtl[label] = dict(e["test_per_task"])

    stl: Dict[str, float] = {}
    for d in sorted(Path(rd).glob(f"stl_task*_n{n_train}_seed{seed}")):
        e = best(rd, d.name)
        if e is None:
            continue
        m = re.match(rf"stl_task\d+_(.+)_n{n_train}_seed{seed}$", d.name)
        if m and m.group(1) in e["test_per_task"]:
            stl[m.group(1)] = e["test_per_task"][m.group(1)]

    keys = set().union(*(set(v) for v in mtl.values())) if mtl else set(stl)
    return dict(mtl=mtl, stl=stl, tasks=[t for t in TASK_NAMES if t in keys],
                dir=rd)


def load_tau_trace(backbone: str, tag: str, method: str = "aim_matrix",
                   n_train: int = N_TRAIN, seed: int = SEED) -> Optional[dict]:
    """Per-epoch tau and policy-loss terms. Free -- no measurement needed."""
    h = history(RESULTS[backbone][tag], f"{method}_n{n_train}_seed{seed}")
    if not h or "tau" not in h[0]:
        return None
    ep, tau, lmag, lprog = [], [], [], []
    for e in h:
        ep.append(e["epoch"])
        tau.append(np.asarray(e["tau"], dtype=float))
        lmag.append(e.get("L_magnitude", np.nan))
        lprog.append(e.get("L_progress", np.nan))
    return dict(epoch=np.array(ep), tau=np.stack(tau) if tau[0].ndim else
                np.array([float(t) for t in tau]),
                l_mag=np.array(lmag, dtype=float),
                l_prog=np.array(lprog, dtype=float),
                best=min(h, key=best_epoch_key))


# ---------------------------------------------------------------------------
# Metrics (mirror metrics.py; no scipy needed for these two)
# ---------------------------------------------------------------------------

def mean_rank(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    methods = list(results)
    tasks = list(next(iter(results.values())))
    ranks = {m: [] for m in methods}
    for t in tasks:
        for r, m in enumerate(sorted(methods,
                                     key=lambda m: results[m].get(t, np.inf)), 1):
            ranks[m].append(r)
    return {m: float(np.mean(v)) for m, v in ranks.items()}


def delta_m_percent(results: Dict[str, Dict[str, float]],
                    stl: Dict[str, float]) -> Dict[str, float]:
    """Mean per-task degradation vs single-task -- LOWER IS BETTER.

        dm% = (1/T) * sum_t 100 * (MAE_method,t - MAE_STL,t) / |MAE_STL,t|

    This is the AIM paper's convention (Table 2: "Lower values are better"),
    so the numbers here line up with the published ones. Note that the repo's
    own metrics.py uses the OPPOSITE sign -- it negates this and calls higher
    better -- so figures from analysis.py / comparison_table.py / the scripts
    at the repo root will disagree in sign with these notebooks. Flip one or
    the other before putting them side by side.

    0 = matches STL. Positive = worse than STL. Negative = beats STL.
    """
    out = {}
    for m, per_task in results.items():
        vals = [100.0 * (per_task[t] - stl[t]) / abs(stl[t])
                for t in stl if t in per_task and abs(stl[t]) > 1e-12]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# Gradient-measurement caches (written by 10_measure_gradients.ipynb)
# ---------------------------------------------------------------------------

def cache_path(tag: str, method: str, split: str = "primary",
               n_train: int = N_TRAIN, seed: int = SEED) -> Path:
    stem = ("cosine_2task" if tag == "2task" else f"cosine_matrix_{tag}")
    return CACHE_DIR / f"{stem}_{method}_{split}_n{n_train}_seed{seed}.csv"


def load_cache(tag: str, method: str, split: str = "primary") -> Dict[str, dict]:
    """{backbone: {'cos': [B,n,n], 'tau': [n,n] or None, 'tasks': [...]}}"""
    p = cache_path(tag, method, split)
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\nRun 10_measure_gradients.ipynb first "
            f"(tag={tag}, method={method}).")
    by: Dict[str, dict] = {}
    with open(p) as fh:
        for row in csv.DictReader(fh):
            tasks = row["tasks"].split("|")
            n = len(tasks)
            raw = row.get("tau") or ""
            tau = (np.array([float(x) for x in raw.split("|")]).reshape(n, n)
                   if raw else None)
            e = by.setdefault(row["backbone"],
                              dict(cos=[], tau=tau, tasks=tasks,
                                   epoch=int(row.get("epoch", -1) or -1)))
            if "matrix" in row:                       # N x N per batch
                e["cos"].append(np.array(
                    [float(x) for x in row["matrix"].split("|")]).reshape(n, n))
            else:                                     # 2-task: one cosine
                c = float(row["cosine"])
                e["cos"].append(np.array([[1.0, c], [c, 1.0]]))
    for e in by.values():
        e["cos"] = np.stack(e["cos"])
    return by


def aim_weight(cos: np.ndarray, tau: np.ndarray, k: float = 10.0) -> np.ndarray:
    """Eq. 1: w_ij = sigma(k (tau_ij - cos phi_ij)), per batch."""
    return 1.0 / (1.0 + np.exp(-k * (tau[None] - cos)))


def heatmap(ax, M, tasks, cm, vmin, vmax, title="", fmt="{:+.2f}",
            annotate=False, title_loc="left"):
    """Square matrix with the diagonal masked (i == j carries no information)."""
    shown = np.array(M, dtype=float).copy()
    np.fill_diagonal(shown, np.nan)
    im = ax.imshow(shown, cmap=cm, vmin=vmin, vmax=vmax)
    n = len(tasks)
    ax.set_xticks(range(n), tasks, rotation=45, ha="right")
    ax.set_yticks(range(n), tasks)
    if title:
        ax.set_title(title, color=INK, loc=title_loc, pad=8)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if annotate:
        hi = max(abs(vmin), abs(vmax))
        for i in range(n):
            for j in range(n):
                if i != j:
                    v = M[i, j]
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=6.5,
                            color=SURFACE if abs(v) > 0.6 * hi else INK)
    return im


# ---------------------------------------------------------------------------
# Table rendering (hairlines only; no boxed grid, no zebra striping)
# ---------------------------------------------------------------------------

def table_png(rows, title, subtitle, note="", highlight=(), fname=None,
              subdir=""):
    """rows[0] is the header. Cells ending in '*' are drawn in the accent
    colour; `highlight` is a set of (row, col) indices to accent as well."""
    n_r, n_c = len(rows), len(rows[0])
    widths = [max(len(str(r[i])) for r in rows) + 2 for i in range(n_c)]
    total = sum(widths)
    row_h, head_h = 0.30, 1.05
    fig_h = head_h + row_h * n_r + 0.30
    fig = plt.figure(figsize=(max(6, 0.105 * total + 0.6), fig_h),
                     facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total); ax.set_ylim(0, fig_h)

    xs, x = [], 0.6
    for w in widths:
        xs.append(x); x += w
    top = fig_h - head_h

    fig.text(0.6 / total, (fig_h - 0.34) / fig_h, title, ha="left",
             va="center", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.6 / total, (fig_h - 0.66) / fig_h, subtitle, ha="left",
             va="center", fontsize=8.5, color=MUTED)

    for ri, row in enumerate(rows):
        y = top - row_h * (ri + 0.5)
        if ri == 0:
            ax.add_patch(plt.Rectangle((0, y - row_h / 2), total, row_h,
                                       facecolor=BAND, edgecolor="none"))
        for ci, cell in enumerate(row):
            cell = str(cell)
            star = cell.endswith("*") or (ri, ci) in highlight
            ax.text(xs[ci] + (0 if ci == 0 else widths[ci] - 2), y,
                    cell.rstrip("*"),
                    ha="left" if ci == 0 else "right", va="center", fontsize=9,
                    color=ACCENT if star else INK,
                    fontweight="bold" if (ri == 0 or star) else "normal")
        ax.plot([0, total], [y - row_h / 2] * 2, color=GRID, lw=0.8, zorder=0)

    ax.plot([0, total], [top] * 2, color=SECONDARY, lw=1.0)
    if note:
        ax.text(0.6, 0.22, note, ha="left", va="center", fontsize=7.5,
                color=MUTED)
    if fname:
        save(fig, fname, subdir)
    return fig
