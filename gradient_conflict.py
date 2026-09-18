"""
Gradient-conflict comparison: Uni-Mol backbone vs MPNN (GNN).

Answers two questions the deck raises but the training code never logged:

  1. How much do task gradients actually conflict on each backbone?
     cos phi_ij = <g_i, g_j> / (||g_i|| ||g_j||),  conflict <=> cos phi_ij < 0

  2. How much of that conflict does AIM actually remove?
     Eq. 1  w_ij = sigma( k (tau_ij - cos phi_ij) )
     Eq. 2  g_i' = g_i - sum_{j != i} w_ij (<g_i,g_j> / ||g_j||^2) g_j
     Eq. 3  g*   = sum_i g_i'
     removal_i = ||g_i - g_i'|| / ||g_i||          (per task)
     shrink    = ||g*|| / ||sum_i g_i||            (whole update)

Both are MEASURED, not read from history.json: `train.py` records tau but never
cos phi_ij, so the conflict structure has to be recomputed from a checkpoint.

Two views, as requested:
  * 2-task  -- the same trained encoder restricted to one task pair (default
               mu / eps_LUMO). This is a restriction of the 11-task model, NOT a
               separately trained 2-task run: no such run exists on disk.
  * 11-task -- the full AIM-paper property set, all 55 unordered pairs.

Usage
-----
    python gradient_conflict.py                          # measure + plot both backbones
    python gradient_conflict.py --backbones gnn          # one backbone only
    python gradient_conflict.py --plot-only              # re-plot from the cache
    python gradient_conflict.py --n_batches 64 --method aim_matrix

Outputs (into --out, default ./conflict_analysis/):
    conflict_stats.json      raw measurements, so plots are reproducible
    conflict_2task.png
    conflict_11task.png
    conflict_summary.png

Notes / caveats
---------------
* Conflict is a property of a MODEL STATE, not of an architecture in the
  abstract. Everything here is measured at the checkpoint named by --method
  (default aim_matrix), on primary-split batches. Say so when you quote it.
* tau comes from the same checkpoint's policy state. With --method ls/pcgrad
  (no policy) tau falls back to 0, which is the AIM initialisation, and the
  "removal" panels then show what AIM would strip at step 1 rather than what a
  trained policy strips.
* Uni-Mol needs `unimol_tools` installed and its runs finished. If a backbone
  cannot be loaded the script says so and plots whatever it could measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent

# Modules that exist under BOTH GNN/src and Unimol/src and would otherwise be
# served from whichever tree was imported first.
_CLASHING = (
    "data", "model", "train", "metrics", "baselines", "aim_optimizer",
    "analysis", "gnn_collate", "unimol_collate",
)

BACKBONES: Dict[str, dict] = {
    "unimol": dict(src="Unimol/src", results="Unimol/results", label="Uni-Mol"),
    "gnn":    dict(src="GNN/src",    results="GNN/result_updated", label="MPNN (GNN)"),
}

TASK_NAMES = ["mu", "alpha", "eps_HOMO", "eps_LUMO", "R2", "zpve",
              "U0", "U", "H", "G", "Cv"]
N_TASKS = 11


# ---------------------------------------------------------------------------
# Isolated per-backbone imports
# ---------------------------------------------------------------------------

def _import_backbone(key: str):
    """Import a backbone's data/model/aim modules with its own src/ on sys.path."""
    src = ROOT / BACKBONES[key]["src"]
    if not src.is_dir():
        raise FileNotFoundError(f"{src} not found")

    for name in _CLASHING:
        sys.modules.pop(name, None)
    sys.path = [p for p in sys.path if "GNN" not in p and "Unimol" not in p]
    sys.path.insert(0, str(src))

    import importlib
    ns = argparse.Namespace()
    ns.data = importlib.import_module("data")
    ns.model = importlib.import_module("model")
    ns.aim = importlib.import_module("aim_optimizer")
    return ns


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _flat_grads(grads, params, torch):
    parts = []
    for g, p in zip(grads, params):
        parts.append(g.reshape(-1) if g is not None else p.new_zeros(p.numel()))
    return torch.cat(parts)


def _load_tau(ckpt: dict, n: int, np_mod) -> tuple:
    """Return (tau [n,n], source string)."""
    pol = ckpt.get("policy")
    if not pol:
        return np_mod.zeros((n, n), dtype=np_mod.float64), "tau = 0 (no policy in checkpoint)"
    t = pol.get("tau")
    if t is None:
        return np_mod.zeros((n, n), dtype=np_mod.float64), "tau = 0 (policy has no tau)"
    arr = np_mod.asarray(t.detach().cpu().numpy(), dtype=np_mod.float64)
    if arr.ndim == 0:
        return np_mod.full((n, n), float(arr)), "learned scalar tau"
    return arr, "learned tau matrix"


def measure_backbone(key: str, args) -> Optional[dict]:
    """Collect per-batch cosine matrices and AIM removal for one backbone."""
    import torch
    import torch.nn.functional as F

    info = BACKBONES[key]
    run = (f"{args.method}_n{args.n_train}_seed{args.seed}")
    ckpt_path = ROOT / info["results"] / run / "best_model.pt"

    print(f"\n[{info['label']}] run={run}")
    if not ckpt_path.exists():
        print(f"  !! no checkpoint at {ckpt_path} -- skipping this backbone")
        return None

    try:
        ns = _import_backbone(key)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! cannot import {info['src']}: {exc} -- skipping")
        return None

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        loaders = ns.data.get_loaders(
            root=str(ROOT / args.data_root), n_train=args.n_train,
            batch_size=args.batch_size, seed=args.seed,
        )
        model = ns.model.build_model(
            device=device, n_tasks=N_TASKS,
            head_hidden=args.head_hidden, trainable_layers=-1,
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! cannot build data/model: {exc} -- skipping")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()                        # deterministic: no dropout in the measurement
                                        # (eval() does not disable autograd)
    tau_full, tau_src = _load_tau(ckpt, N_TASKS, np)
    print(f"  checkpoint epoch {ckpt.get('epoch', '?')} | {tau_src}")

    means = loaders["means"].to(device)
    stds = loaders["stds"].to(device)
    shared = model.shared_params()
    d = sum(p.numel() for p in shared)
    print(f"  shared trainable dim d = {d:,}")

    cos_batches: List[np.ndarray] = []
    norm_batches: List[np.ndarray] = []

    for b, batch in enumerate(loaders["primary_loader"]):
        if b >= args.n_batches:
            break
        batch = ns.data.batch_to_device(batch, device)
        y = (batch["targets"] - means) / stds
        h = model.shared_forward(batch)

        gs = []
        for i in range(N_TASKS):
            loss_i = F.l1_loss(model.task_forward(h, i), y[:, i])
            gi = torch.autograd.grad(
                loss_i, shared, retain_graph=(i < N_TASKS - 1), allow_unused=True)
            gs.append(_flat_grads(gi, shared, torch))

        G = torch.stack(gs).double()                       # [11, d]
        Gn = F.normalize(G, dim=1, eps=1e-12)
        cos_batches.append((Gn @ Gn.t()).cpu().numpy())
        norm_batches.append(G.norm(dim=1).cpu().numpy())
        del G, Gn, gs, h
        if (b + 1) % 10 == 0:
            print(f"    batch {b + 1}/{args.n_batches}")

    if not cos_batches:
        print("  !! no batches measured -- skipping")
        return None

    print(f"  measured {len(cos_batches)} batches")
    return dict(
        key=key, label=info["label"], run=run,
        epoch=int(ckpt.get("epoch", -1)), tau_source=tau_src, d=int(d),
        tau=tau_full.tolist(),
        cos=[c.tolist() for c in cos_batches],
        norms=[n.tolist() for n in norm_batches],
    )


# ---------------------------------------------------------------------------
# Derived quantities (pure numpy -- no torch needed, so --plot-only is cheap)
# ---------------------------------------------------------------------------

def aim_view(rec: dict, idx: List[int], k: float) -> dict:
    """Restrict a measurement to `idx` tasks and apply Eq. 1-3 with the learned tau."""
    ii = np.ix_(idx, idx)
    n = len(idx)
    eye = np.eye(n)
    tau = np.asarray(rec["tau"])[ii]

    cos_all, removal_all, shrink_all, gate_all = [], [], [], []
    for C_full, nrm_full in zip(rec["cos"], rec["norms"]):
        C = np.asarray(C_full)[ii]
        nrm = np.asarray(nrm_full)[idx]

        # Eq. 1 gate
        w = 1.0 / (1.0 + np.exp(-k * (tau - C)))

        # Eq. 2: g_i' = g_i - sum_j w_ij (g_i.g_j / ||g_j||^2) g_j, in Gram form.
        # dot_ij = C_ij ||g_i|| ||g_j||  ->  coef_ij = C_ij ||g_i|| / ||g_j||
        outer = np.outer(nrm, 1.0 / np.maximum(nrm, 1e-12))
        coef = C * outer
        M = np.eye(n) - (w * coef * (1.0 - eye))          # G' = M G

        # ||g_i - g_i'||^2 = (M-I)_i Gram (M-I)_i^T  with Gram_ij = C_ij ||g_i|| ||g_j||
        gram = C * np.outer(nrm, nrm)
        D = M - np.eye(n)
        removed_sq = np.einsum("ij,jk,ik->i", D, gram, D)
        removal = np.sqrt(np.maximum(removed_sq, 0.0)) / np.maximum(nrm, 1e-12)

        ones = np.ones(n)
        aim_sq = ones @ M @ gram @ M.T @ ones
        sum_sq = ones @ gram @ ones
        shrink = np.sqrt(max(aim_sq, 0.0)) / max(np.sqrt(max(sum_sq, 1e-24)), 1e-12)

        cos_all.append(C)
        removal_all.append(removal)
        shrink_all.append(shrink)
        gate_all.append(w[~np.eye(n, dtype=bool)])

    cos_all = np.stack(cos_all)                            # [B, n, n]
    off = ~np.eye(n, dtype=bool)
    pair_vals = cos_all[:, off].ravel()

    per_task_conflict = np.array([
        float((cos_all[:, i, off[i]] < 0).mean()) for i in range(n)
    ])

    return dict(
        label=rec["label"], key=rec["key"], n=n, idx=idx,
        names=[TASK_NAMES[i] for i in idx],
        cos_mean=cos_all.mean(axis=0),
        pair_vals=pair_vals,
        conflict_frac=float((pair_vals < 0).mean()),
        per_task_conflict=per_task_conflict,
        removal=np.stack(removal_all).mean(axis=0),
        removal_overall=float(np.stack(removal_all).mean()),
        shrink=float(np.mean(shrink_all)),
        gate_mean=float(np.mean(np.concatenate(gate_all))),
        tau=np.asarray(rec["tau"])[ii],
        epoch=rec["epoch"], tau_source=rec["tau_source"],
    )


# ---------------------------------------------------------------------------
# Plotting -- palette and chrome from the validated reference instance
# ---------------------------------------------------------------------------

SERIES = {"unimol": "#2a78d6", "gnn": "#eb6834"}   # categorical slots 1, 2
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Diverging: red (conflict) <-> neutral gray <-> blue (aligned). Equal arms.
DIVERGING = ["#96201f", "#d03b3b", "#f3a09f", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"]


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 140,
    })


def _clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("conflict", DIVERGING)


def _panel_distribution(ax, views):
    """Where the pairwise cosines actually sit, per backbone."""
    lo = min(float(v["pair_vals"].min()) for v in views)
    hi = max(float(v["pair_vals"].max()) for v in views)
    pad = 0.05 * max(hi - lo, 1e-3)
    bins = np.linspace(lo - pad, hi + pad, 41)

    if bins[0] < 0:
        ax.axvspan(bins[0], 0, color="#f0efec", zorder=0)
        ax.text(bins[0], 0.02, "  conflict", color=MUTED, fontsize=8,
                va="bottom", ha="left", transform=ax.get_xaxis_transform())

    for v in views:
        ax.hist(v["pair_vals"], bins=bins, density=True, histtype="step",
                linewidth=2, color=SERIES[v["key"]],
                label=f"{v['label']}  ({v['conflict_frac'] * 100:.0f}% conflicting)",
                zorder=3)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
    _clean(ax)
    ax.set_xlabel("cos $\\varphi_{ij}$  between task gradients")
    ax.set_ylabel("density")
    ax.margins(y=0.20)
    ax.set_title("Where the pairwise cosines sit", color=INK, loc="left", pad=8)
    ax.legend(loc="upper right")


def _panel_removal(ax, views):
    """How much of each task's gradient AIM projects out."""
    names = views[0]["names"]
    n = len(names)
    x = np.arange(n)
    width = 0.38 if len(views) > 1 else 0.55

    for s, v in enumerate(views):
        off = (s - (len(views) - 1) / 2) * width
        ax.bar(x + off, v["removal"] * 100, width * 0.92,
               color=SERIES[v["key"]], edgecolor=SURFACE, linewidth=1.5,
               label=f"{v['label']}  (mean {v['removal_overall'] * 100:.1f}%)",
               zorder=3)

    _clean(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45 if n > 4 else 0,
                       ha="right" if n > 4 else "center")
    ax.set_ylabel("removed by AIM   $\\|g_i - g_i'\\| \\, / \\, \\|g_i\\|$   (%)")
    ax.margins(y=0.22)
    ax.set_title("How much of each task's gradient AIM projects out",
                 color=INK, loc="left", pad=8)
    ax.legend(loc="upper center", ncol=len(views))


def _panel_heatmaps(fig, gs_row, views, vmax):
    """Mean cos phi_ij, one map per backbone, shared diverging scale."""
    import matplotlib.pyplot as plt
    axes = []
    for s, v in enumerate(views):
        ax = fig.add_subplot(gs_row[s])
        names = v["names"]
        n = v["n"]
        shown = np.array(v["cos_mean"], dtype=float)
        np.fill_diagonal(shown, np.nan)          # cos(g_i, g_i) = 1 always
        cmap = _cmap()
        cmap.set_bad(SURFACE)
        im = ax.imshow(shown, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n), names, rotation=45 if n > 4 else 0,
                      ha="right" if n > 4 else "center")
        ax.set_yticks(range(n), names)
        ax.set_title(f"mean cos $\\varphi_{{ij}}$ — {v['label']}",
                     color=INK, loc="left", pad=8)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        if n <= 4:                      # only label cells when they fit
            for i in range(n):
                for j in range(n):
                    val = v["cos_mean"][i, j]
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=9,
                            color=SURFACE if abs(val) > 0.55 * vmax else INK)
        axes.append((ax, im))

    cb = fig.colorbar(axes[-1][1], ax=[a for a, _ in axes],
                      fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED, labelsize=8)
    cb.set_label("conflict  ←     0     →  aligned", color=SECONDARY, fontsize=8.5)
    return axes


def _panel_pairstrip(ax, views):
    """Few enough pairs that a heatmap would be one number dressed up as a grid."""
    rng = np.random.default_rng(0)
    for s, v in enumerate(views):
        y = len(views) - 1 - s
        vals = v["pair_vals"]
        jit = (rng.random(vals.size) - 0.5) * 0.22
        ax.scatter(vals, np.full(vals.size, y, dtype=float) + jit, s=16,
                   color=SERIES[v["key"]], alpha=0.35, linewidths=0, zorder=3)
        m = float(vals.mean())
        ax.scatter([m], [y], s=110, color=SERIES[v["key"]],
                   edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.text(m, y + 0.34, f"mean {m:+.3f}", color=SECONDARY, fontsize=9,
                ha="center", va="bottom")

    lo, hi = ax.get_xlim()
    if lo < 0:
        ax.axvspan(lo, 0, color="#f0efec", zorder=0)
        ax.text(lo, 0.02, "  conflict", color=MUTED, fontsize=8, va="bottom",
                ha="left", transform=ax.get_xaxis_transform())
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(views)), [v["label"] for v in views][::-1])
    ax.set_ylim(-0.6, len(views) - 0.4)
    _clean(ax, grid_axis="x")
    ax.set_xlabel("cos $\\varphi_{ij}$  per batch")
    ax.set_title("Every batch, one point — the pair's conflict and its spread",
                 color=INK, loc="left", pad=8)


def plot_view(views, title, subtitle, out_png):
    import matplotlib.pyplot as plt
    _style()
    compact = views[0]["n"] <= 3
    fig = plt.figure(figsize=(13.2, 7.0 if compact else 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62] if compact else [1.0, 1.15],
                          hspace=0.45, wspace=0.28,
                          left=0.07, right=0.93, top=0.86, bottom=0.15)

    fig.text(0.07, 0.955, title, fontsize=15, color=INK, fontweight="bold", ha="left")
    fig.text(0.07, 0.915, subtitle, fontsize=9.5, color=MUTED, ha="left")

    _panel_distribution(fig.add_subplot(gs[0, 0]), views)
    _panel_removal(fig.add_subplot(gs[0, 1]), views)

    if compact:
        _panel_pairstrip(fig.add_subplot(gs[1, :]), views)
    else:
        vmax = max(float(np.abs(v["cos_mean"] - np.eye(v["n"])).max()) for v in views)
        _panel_heatmaps(fig, [gs[1, 0], gs[1, 1]], views, max(vmax, 0.05))

    foot = "   ·   ".join(
        f"{v['label']}: epoch {v['epoch']}, {v['tau_source']}, "
        f"‖g*‖/‖Σg‖ = {v['shrink']:.2f}" for v in views)
    fig.text(0.07, 0.015, foot, fontsize=8, color=MUTED, ha="left")

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


def plot_summary(by_view, out_png):
    """2-task vs 11-task, both backbones, three headline metrics."""
    import matplotlib.pyplot as plt
    _style()
    metrics = [
        ("conflict_frac", "Pairs in conflict  (cos $\\varphi_{ij} < 0$)", 100.0, "%"),
        ("removal_overall", "Mean gradient removed by AIM", 100.0, "%"),
        ("shrink", "Update magnitude   $\\|g^*\\| / \\|\\Sigma_i g_i\\|$", 1.0, ""),
    ]
    labels = list(by_view.keys())                      # ["2 tasks", "11 tasks"]
    keys = [v["key"] for v in next(iter(by_view.values()))]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.14, wspace=0.28)
    fig.text(0.06, 0.93, "Conflict and intervention: 2 tasks vs 11 tasks",
             fontsize=15, color=INK, fontweight="bold", ha="left")
    fig.text(0.06, 0.86,
             "Same trained encoders, same batches — only the number of tasks "
             "sharing them changes.", fontsize=9.5, color=MUTED, ha="left")

    x = np.arange(len(labels))
    width = 0.38
    for ax, (field, title, scale, unit) in zip(axes, metrics):
        for s, k in enumerate(keys):
            vals = [by_view[lab][s][field] * scale for lab in labels]
            off = (s - (len(keys) - 1) / 2) * width
            ax.bar(x + off, vals, width * 0.92, color=SERIES[k],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=by_view[labels[0]][s]["label"])
            for xi, val in zip(x + off, vals):
                ax.text(xi, val, f"{val:.0f}{unit}" if scale == 100 else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8.5, color=SECONDARY)
        if field == "shrink":
            ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=2)
            ax.text(ax.get_xlim()[1], 1.0, " unchanged", color=MUTED, fontsize=8,
                    va="center", ha="left")
        _clean(ax)
        ax.set_xticks(x, labels)
        ax.set_title(title, color=INK, loc="left", pad=8, fontsize=10)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(keys)], names[:len(keys)], loc="upper right",
               bbox_to_anchor=(0.97, 0.99), ncol=len(keys))

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Measure and plot gradient conflict + AIM intervention, "
                    "Uni-Mol vs MPNN, for 2 tasks and for 11 tasks.")
    p.add_argument("--backbones", nargs="+", default=["unimol", "gnn"],
                   choices=list(BACKBONES))
    p.add_argument("--method", default="aim_matrix",
                   help="run whose checkpoint (and tau) is measured")
    p.add_argument("--n_train", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--head_hidden", type=int, default=64)
    p.add_argument("--n_batches", type=int, default=32,
                   help="primary batches to average the cosine matrix over")
    p.add_argument("--k", type=float, default=10.0, help="AIM gate temperature")
    p.add_argument("--tasks2", nargs=2, type=int, default=[0, 3],
                   metavar=("I", "J"),
                   help=f"the 2-task view, indices into {TASK_NAMES}")
    p.add_argument("--data_root", default="data/qm9")
    p.add_argument("--out", default="conflict_analysis")
    p.add_argument("--device", default=None)
    p.add_argument("--plot-only", action="store_true",
                   help="skip measurement, re-plot from conflict_stats.json")
    args = p.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "conflict_stats.json"

    if args.plot_only:
        if not cache.exists():
            sys.exit(f"no cache at {cache} -- run without --plot-only first")
        records = json.loads(cache.read_text())
        print(f"loaded {len(records)} backbone(s) from {cache}")
    else:
        records = [r for r in (measure_backbone(k, args) for k in args.backbones)
                   if r is not None]
        if not records:
            sys.exit("\nNothing measured. Check --data_root, the checkpoints, "
                     "and that the backbone's dependencies are installed.")
        cache.write_text(json.dumps(records))
        print("\nwrote", cache)

    two = [TASK_NAMES[i] for i in args.tasks2]
    views_2 = [aim_view(r, list(args.tasks2), args.k) for r in records]
    views_11 = [aim_view(r, list(range(N_TASKS)), args.k) for r in records]

    plot_view(
        views_2,
        f"Gradient conflict on 2 tasks — {two[0]} vs {two[1]}",
        "The 11-task encoders restricted to one task pair "
        "(no 2-task run exists on disk). Measured on primary-split batches.",
        out / "conflict_2task.png")

    plot_view(
        views_11,
        "Gradient conflict on all 11 tasks",
        "Full AIM-paper property set — 55 unordered pairs per batch. "
        "Measured on primary-split batches.",
        out / "conflict_11task.png")

    plot_summary({"2 tasks": views_2, "11 tasks": views_11},
                 out / "conflict_summary.png")

    print("\nHeadline numbers")
    for lab, views in (("2 tasks", views_2), ("11 tasks", views_11)):
        for v in views:
            print(f"  {lab:9s} {v['label']:12s} "
                  f"conflicting {v['conflict_frac'] * 100:5.1f}%   "
                  f"AIM removes {v['removal_overall'] * 100:5.1f}%   "
                  f"gate {v['gate_mean']:.2f}   shrink {v['shrink']:.2f}")


if __name__ == "__main__":
    main()
