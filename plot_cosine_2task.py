"""
Task-gradient geometry on the 2-task setting -- Uni-Mol vs GNN.

Question: does the backbone change task-gradient geometry?

With two tasks there is exactly one task pair, so the cosine "matrix" is a
single number per backbone -- a 2x2 heatmap would render one value twice. What
is actually informative is the DISTRIBUTION of

    cos(g_mu, g_eps_LUMO) = <g_i, g_j> / (||g_i|| ||g_j||)

across batches: it shows whether the gap between the two backbones is larger
than the batch-to-batch spread, which a single averaged cell hides. Keep the
heatmap form for the 11-task set, where the grid has real structure.

This has to be MEASURED. train.py never logs cos(g_i, g_j), so the script loads
a checkpoint, runs one forward and two backward passes per batch, and builds
the cosine directly. No re-training -- but it does compute gradients, which is
the only way these numbers can exist.

Which checkpoint: LS by default, not AIM. AIM's weights were reached by
modifying gradients, so its geometry is partly the method's doing; LS is plain
joint training and isolates the backbone.

Usage (run it in the env that has unimol_tools -- e.g. conda env aim_gnn)
------------------------------------------------------------------------
    python plot_cosine_2task.py
    python plot_cosine_2task.py --method aim_matrix --split test
    python plot_cosine_2task.py --n_batches 32
    python plot_cosine_2task.py --backbones gnn        # one backbone only
    python plot_cosine_2task.py --force                # ignore the cache

Saves into figures/:
    cosine_2task_<method>_<split>_n<N>_seed<S>.png
    cosine_2task_<method>_<split>_n<N>_seed<S>.csv   per-batch cosines (cache)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent

_CLASHING = ("data", "model", "train", "metrics", "baselines", "aim_optimizer",
             "analysis", "gnn_collate", "unimol_collate")

BACKBONES: Dict[str, dict] = {
    "Uni-Mol": dict(src="Unimol/src", label="Uni-Mol",
                    trees=["../Results/Unimol/results_2task",
                           "Unimol/results_2task",
                           "Unimol/results/tasks_mu-eps_LUMO"]),
    "GNN": dict(src="GNN/src", label="GNN",
                trees=["../Results/GNN/result_2task",
                       "GNN/result_2task",
                       "GNN/result_updated/tasks_mu-eps_LUMO"]),
}
KEY = {"unimol": "Uni-Mol", "gnn": "GNN"}

SERIES = {"Uni-Mol": "#2a78d6", "GNN": "#eb6834"}
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"


# ---------------------------------------------------------------------------

def _import_backbone(src_rel: str):
    src = ROOT / src_rel
    if not src.is_dir():
        raise FileNotFoundError(src)
    for name in _CLASHING:
        sys.modules.pop(name, None)
    sys.path = [p for p in sys.path if "GNN" not in p and "Unimol" not in p]
    sys.path.insert(0, str(src))
    import importlib
    ns = argparse.Namespace()
    ns.data = importlib.import_module("data")
    ns.model = importlib.import_module("model")
    return ns


def _find_tree(cfg: dict, method: str, n_train: int, seed: int) -> Optional[Path]:
    for rel in cfg["trees"]:
        p = ROOT / rel
        if (p / f"{method}_n{n_train}_seed{seed}" / "best_model.pt").exists():
            return p
    return None


def measure(name: str, args) -> Optional[dict]:
    import torch
    import torch.nn.functional as F

    cfg = BACKBONES[name]
    tree = _find_tree(cfg, args.method, args.n_train, args.seed)
    if tree is None:
        print(f"  !! {name}: no {args.method} checkpoint in any of "
              + ", ".join(cfg["trees"]))
        return None
    ckpt_path = tree / f"{args.method}_n{args.n_train}_seed{args.seed}" / "best_model.pt"
    print(f"\n[{name}] {ckpt_path}")

    try:
        ns = _import_backbone(cfg["src"])
    except Exception as exc:                                    # noqa: BLE001
        print(f"  !! cannot import {cfg['src']}: {exc}")
        return None

    cols = ns.data.parse_task_spec(args.tasks)
    names = [ns.data.TASK_NAMES[i] for i in cols]
    if len(cols) != 2:
        print(f"  !! expected 2 tasks, got {names}")
        return None

    device = torch.device(args.device or "cpu")
    try:
        loaders = ns.data.get_loaders(
            root=str(ROOT / args.data_root), n_train=args.n_train,
            batch_size=args.batch_size, seed=args.seed, target_cols=cols)
        model = ns.model.build_model(device=device, n_tasks=2,
                                     head_hidden=args.head_hidden,
                                     trainable_layers=-1)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  !! cannot build data/model: {exc}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()                       # deterministic; autograd still active
    print(f"  checkpoint epoch {ckpt.get('epoch', '?')}  tasks {names}")

    means = loaders["means"].to(device)
    stds = loaders["stds"].to(device)
    shared = model.shared_params()
    print(f"  shared trainable dim d = {sum(p.numel() for p in shared):,}")

    loader = loaders[f"{args.split}_loader"]
    cosines: List[float] = []
    for b, batch in enumerate(loader):
        if b >= args.n_batches:
            break
        batch = ns.data.batch_to_device(batch, device)
        y = (batch["targets"] - means) / stds
        h = model.shared_forward(batch)

        gs = []
        for i in range(2):
            loss_i = F.l1_loss(model.task_forward(h, i), y[:, i])
            gi = torch.autograd.grad(loss_i, shared, retain_graph=(i == 0),
                                     allow_unused=True)
            gs.append(torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(gi, shared)]).double())

        c = F.cosine_similarity(gs[0], gs[1], dim=0, eps=1e-12).item()
        cosines.append(c)
        if (b + 1) % 4 == 0:
            print(f"    batch {b + 1}/{args.n_batches}  cos={c:+.4f}")

    if not cosines:
        print("  !! no batches measured")
        return None
    pol = ckpt.get("policy") or {}
    t = pol.get("tau")
    if t is None:
        tau = None
    else:
        a = np.asarray(t.detach().cpu().numpy(), dtype=float)
        tau = np.full((2, 2), float(a)) if a.ndim == 0 else a

    return dict(name=name, cos=np.array(cosines), tasks=names, tau=tau,
                epoch=int(ckpt.get("epoch", -1)), tree=str(tree))


# ---------------------------------------------------------------------------

def _style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.labelsize": 9.5, "ytick.labelsize": 11,
        "legend.frameon": False, "figure.dpi": 140,
    })


def make_figure(results: List[dict], args) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    rng = np.random.default_rng(0)
    order = [r for n in SERIES for r in results if r["name"] == n]

    for s, r in enumerate(order):
        y = len(order) - 1 - s
        v, c = r["cos"], SERIES[r["name"]]
        ax.scatter(v, np.full(v.size, y) + (rng.random(v.size) - 0.5) * 0.26,
                   s=26, color=c, alpha=0.40, linewidths=0, zorder=3)
        m, sd = float(v.mean()), float(v.std())
        ax.plot([m - sd, m + sd], [y, y], color=c, lw=2.5, alpha=0.55, zorder=4)
        ax.scatter([m], [y], s=150, color=c, edgecolor=SURFACE, lw=2, zorder=5)
        ax.text(m, y + 0.30,
                f"mean {m:+.3f}  ±{sd:.3f}   ·   "
                f"{100 * (v < 0).mean():.0f}% of batches conflicting",
                color=SECONDARY, fontsize=9.5, ha="center", va="bottom")

    lo, hi = ax.get_xlim()
    if lo < 0:
        ax.axvspan(lo, 0, color=BAND, zorder=0)
        ax.text(lo, 0.03, "  conflict  (cos < 0)", color=MUTED, fontsize=8.5,
                va="bottom", ha="left", transform=ax.get_xaxis_transform())
    ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xlim(lo, hi)

    ax.set_yticks(range(len(order)), [r["name"] for r in order][::-1])
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.set_xlabel(f"cos $\\varphi$  between the {order[0]['tasks'][0]} and "
                  f"{order[0]['tasks'][1]} gradients,  one point per batch")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Task-gradient geometry: does the backbone change it?",
                 x=0.012, y=0.98, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.012, 0.885,
             f"2-task setting ({' / '.join(order[0]['tasks'])}) · measured at the "
             f"{args.method.upper()} checkpoint on the {args.split} split · "
             f"{args.n_batches} batches of {args.batch_size} · "
             f"n_train={args.n_train}, seed {args.seed}",
             fontsize=9.5, color=MUTED, ha="left")
    fig.text(0.012, 0.03,
             "LS chosen over AIM on purpose: AIM's weights were reached by modifying "
             "gradients, so its geometry is partly the method's doing.   "
             + "   ·   ".join(f"{r['name']}: epoch {r['epoch']}" for r in order),
             fontsize=7.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.70, bottom=0.26)
    return fig


def save_csv(path: Path, results: List[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "batch", "cosine", "epoch", "tasks", "tau"])
        for r in results:
            tasks = "|".join(r["tasks"])
            tau = r.get("tau")
            tau_s = "" if tau is None else "|".join(f"{v:.6f}"
                                                   for v in np.ravel(tau))
            for i, c in enumerate(r["cos"]):
                w.writerow([r["name"], i, f"{c:.6f}", r["epoch"], tasks, tau_s])


def load_csv(path: Path) -> List[dict]:
    by: Dict[str, dict] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            raw = row.get("tau") or ""
            tau = (np.array([float(x) for x in raw.split("|")]).reshape(2, 2)
                   if raw else None)
            e = by.setdefault(row["backbone"], dict(
                name=row["backbone"], cos=[], tau=tau,
                epoch=int(row.get("epoch", -1) or -1),
                tasks=(row.get("tasks") or "mu|eps_LUMO").split("|"),
                tree="(cache)"))
            e["cos"].append(float(row["cosine"]))
    for e in by.values():
        e["cos"] = np.array(e["cos"])
    return list(by.values())


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-batch task-gradient cosine, 2-task, Uni-Mol vs GNN.")
    ap.add_argument("--backbones", nargs="+", default=["unimol", "gnn"],
                    choices=list(KEY))
    ap.add_argument("--method", default="ls",
                    help="checkpoint to measure at (ls | pcgrad | aim_matrix | aim_scalar)")
    ap.add_argument("--split", default="primary",
                    choices=["primary", "guidance", "val", "test"])
    ap.add_argument("--tasks", nargs="+", default=["mu", "eps_LUMO"])
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=64)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--data_root", default="data/qm9")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--cache_dir", default="figures",
                    help="where the measurement cache lives; keep it fixed "
                         "so moving --out never forces a re-measure")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--measure-only", action="store_true",
                    help="measure and cache the cosines, skip plotting")
    ap.add_argument("--plot-only", action="store_true",
                    help="plot from the cached CSV, measure nothing")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stem = (f"cosine_2task_{args.method}_{args.split}_"
            f"n{args.n_train}_seed{args.seed}")
    cdir = ROOT / args.cache_dir
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"{stem}.csv"

    if args.plot_only:
        if not cache.exists():
            sys.exit(f"no cache at {cache} -- run --measure-only first")
        print(f"plotting from {cache}")
        results = load_csv(cache)
    else:
        # Cache per backbone, so the two halves can be measured in separate runs
        # (Uni-Mol needs unimol_tools; the GNN half does not).
        have = {r["name"]: r for r in load_csv(cache)} if cache.exists() else {}
        wanted = [KEY[k] for k in args.backbones]
        if args.force:
            have = {k: v for k, v in have.items() if k not in wanted}

        for name in wanted:
            if name in have:
                print(f"[{name}] already cached -- skipping (--force to re-measure)")
                continue
            r = measure(name, args)
            if r is not None:
                have[name] = r

        results = list(have.values())
        if not results:
            sys.exit("\nNothing measured. Check the checkpoints, --data_root, "
                     "and that this interpreter has the backbone's deps "
                     "(unimol_tools for Uni-Mol).")
        save_csv(cache, results)
        print(f"\nsaved {cache}")

    print()
    for r in results:
        v = r["cos"]
        print(f"  {r['name']:9s} cos = {v.mean():+.4f} ± {v.std():.4f}   "
              f"conflicting in {100 * (v < 0).mean():.0f}% of {v.size} batches")

    if args.measure_only:
        print("\n--measure-only: cosines cached, nothing plotted.\n"
              "Plot them from an interpreter with a working matplotlib:\n"
              f"    python {Path(__file__).name} --plot-only")
        return

    fig = make_figure(results, args)
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    print(f"\nsaved {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
