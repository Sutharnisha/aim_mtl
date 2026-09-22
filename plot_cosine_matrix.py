"""
Average gradient cosine similarity -- Uni-Mol vs GNN, full task set.

Question: does the backbone change task-gradient geometry?

    cos phi_ij = <g_i, g_j> / (||g_i|| ||g_j||)

averaged over batches, as an N x N heatmap per backbone, plus the difference
map that shows WHERE the two backbones disagree. With eleven tasks the grid has
real structure (the U0/U/H/G energy block, mu against everything else), which is
why a heatmap earns its place here -- for two tasks use plot_cosine_2task.py,
whose strip plot shows the batch spread a single averaged cell would hide.

MEASURED, not logged: train.py discards the per-step cosines, so this loads a
checkpoint and runs one forward + N backward passes per batch. No re-training.

Which checkpoint: LS by default. AIM's weights were reached by modifying
gradients, so its geometry is partly the method's doing; LS is plain joint
training and isolates the backbone, which is what the question asks about.

Two phases, because matplotlib crashes inside the conda env that carries
unimol_tools:

    # measure  (env with torch + rdkit + unimol_tools)
    <aim_gnn python> plot_cosine_matrix.py --measure-only
    # plot     (env with a working matplotlib)
    python plot_cosine_matrix.py --plot-only

Saves into figures/:
    cosine_matrix_<tasks>task_<method>_<split>_n<N>_seed<S>.png
    cosine_matrix_<tasks>task_<method>_<split>_n<N>_seed<S>.csv   (cache)

Memory note: N gradient vectors over the shared parameters are held at once --
11 x 47.3 M in fp32 is ~2.1 GB for Uni-Mol. fp32 throughout for that reason;
the GNN side is trivial at 1.72 M.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_cosine_2task import _import_backbone  # noqa: E402

KEY = {"unimol": "Uni-Mol", "gnn": "GNN"}
TREES = {
    "Uni-Mol": dict(src="Unimol/src",
                    trees=["../Results/Unimol/results_11task",
                           "Unimol/results_11task", "Unimol/results"]),
    "GNN": dict(src="GNN/src",
                trees=["../Results/GNN/result_11task",
                       "GNN/result_11task", "GNN/result_updated"]),
}

SERIES = {"Uni-Mol": "#2a78d6", "GNN": "#eb6834"}
SURFACE, INK, SECONDARY, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
# Diverging: red (conflict) <-> neutral gray <-> blue (aligned). Equal arms.
DIVERGING = ["#96201f", "#d03b3b", "#f3a09f", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"]


def _style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "figure.dpi": 140,
    })


# ---------------------------------------------------------------------------

def stem_tail(args) -> str:
    n = len(args.tasks) if args.tasks else 11
    return (f"{n}task_{args.method}_{args.split}_"
            f"n{args.n_train}_seed{args.seed}")


def _find_tree(cfg: dict, method: str, n_train: int, seed: int) -> Optional[Path]:
    for rel in cfg["trees"]:
        p = ROOT / rel
        if (p / f"{method}_n{n_train}_seed{seed}" / "best_model.pt").exists():
            return p
    return None


def measure(name: str, args) -> Optional[dict]:
    import torch
    import torch.nn.functional as F

    cfg = TREES[name]
    tree = _find_tree(cfg, args.method, args.n_train, args.seed)
    if tree is None:
        print(f"  !! {name}: no {args.method} checkpoint in "
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
    n = len(cols)

    device = torch.device(args.device or "cpu")
    try:
        loaders = ns.data.get_loaders(
            root=str(ROOT / args.data_root), n_train=args.n_train,
            batch_size=args.batch_size, seed=args.seed, target_cols=cols)
        model = ns.model.build_model(device=device, n_tasks=n,
                                     head_hidden=args.head_hidden,
                                     trainable_layers=-1)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  !! cannot build data/model: {exc}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()                       # deterministic; autograd stays active
    print(f"  checkpoint epoch {ckpt.get('epoch', '?')}  {n} tasks")

    means = loaders["means"].to(device)
    stds = loaders["stds"].to(device)
    shared = model.shared_params()
    d = sum(p.numel() for p in shared)
    print(f"  shared trainable dim d = {d:,}   "
          f"(~{n * d * 4 / 1e9:.2f} GB for {n} fp32 gradients)")

    loader = loaders[f"{args.split}_loader"]
    mats: List[np.ndarray] = []
    for b, batch in enumerate(loader):
        if b >= args.n_batches:
            break
        batch = ns.data.batch_to_device(batch, device)
        y = (batch["targets"] - means) / stds
        h = model.shared_forward(batch)

        rows = []
        for i in range(n):
            loss_i = F.l1_loss(model.task_forward(h, i), y[:, i])
            gi = torch.autograd.grad(loss_i, shared, retain_graph=(i < n - 1),
                                     allow_unused=True)
            flat = torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(gi, shared)]).float()
            rows.append(F.normalize(flat, dim=0, eps=1e-12))
        G = torch.stack(rows)                       # [n, d], unit rows
        mats.append((G @ G.t()).cpu().numpy())
        del G, rows, h
        print(f"    batch {b + 1}/{args.n_batches}")

    if not mats:
        print("  !! no batches measured")
        return None
    pol = ckpt.get("policy") or {}
    t = pol.get("tau")
    if t is None:
        tau = None
    else:
        a = np.asarray(t.detach().cpu().numpy(), dtype=float)
        tau = np.full((n, n), float(a)) if a.ndim == 0 else a

    return dict(name=name, cos=np.stack(mats), tasks=names, tau=tau,
                epoch=int(ckpt.get("epoch", -1)), tree=str(tree))


def save_csv(path: Path, results: List[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "batch", "epoch", "tasks", "matrix", "tau"])
        for r in results:
            tasks = "|".join(r["tasks"])
            tau = r.get("tau")
            tau_s = "" if tau is None else "|".join(f"{v:.6f}"
                                                   for v in np.ravel(tau))
            for b, M in enumerate(r["cos"]):
                w.writerow([r["name"], b, r["epoch"], tasks,
                            "|".join(f"{v:.6f}" for v in M.ravel()), tau_s])


def load_csv(path: Path) -> List[dict]:
    by: Dict[str, dict] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            n0 = len(row["tasks"].split("|"))
            raw = row.get("tau") or ""
            tau = (np.array([float(x) for x in raw.split("|")]).reshape(n0, n0)
                   if raw else None)
            e = by.setdefault(row["backbone"], dict(
                name=row["backbone"], cos=[], epoch=int(row["epoch"]), tau=tau,
                tasks=row["tasks"].split("|"), tree="(cache)"))
            n = len(e["tasks"])
            e["cos"].append(
                np.array([float(x) for x in row["matrix"].split("|")]).reshape(n, n))
    for e in by.values():
        e["cos"] = np.stack(e["cos"])
    return list(by.values())


# ---------------------------------------------------------------------------

def _heat(ax, M, tasks, cmap, vmin, vmax, title, annotate=False, fmt="{:+.2f}"):
    shown = M.astype(float).copy()
    np.fill_diagonal(shown, np.nan)            # i == j carries no information
    cmap.set_bad(SURFACE)
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    n = len(tasks)
    ax.set_xticks(range(n), tasks, rotation=45, ha="right")
    ax.set_yticks(range(n), tasks)
    ax.set_title(title, color=INK, loc="left", pad=8, fontsize=11)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if annotate:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = M[i, j]
                hi = max(abs(vmin), abs(vmax))
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=6.5,
                        color=SURFACE if abs(v) > 0.6 * hi else INK)
    return im


SEQUENTIAL = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#104281"]


def panels(results: List[dict], args, kind: str):
    """{backbone: matrix} for 'cos' | 'freq' | 'weight', in fixed series order."""
    order = [r for n in SERIES for r in results if r["name"] == n]
    out = {}
    for r in order:
        if kind == "cos":
            out[r["name"]] = r["cos"].mean(axis=0)
        elif kind == "freq":
            out[r["name"]] = (r["cos"] < 0).mean(axis=0)
        else:                                     # w_ij, Eq. 1
            tau = r["tau"]
            w = 1.0 / (1.0 + np.exp(-args.k * (tau[None] - r["cos"])))
            out[r["name"]] = w.mean(axis=0)
    return order, out


def make_figure(results: List[dict], args, kind: str = "cos") -> plt.Figure:
    _style()
    order, means = panels(results, args, kind)
    tasks = order[0]["tasks"]
    off = ~np.eye(len(tasks), dtype=bool)

    if kind == "cos":
        cmap = LinearSegmentedColormap.from_list("conflict", DIVERGING)
        vmax = max(float(np.abs(M[off]).max()) for M in means.values())
        vmin = -vmax
        cb_label = r"conflict  ←   mean cos $\varphi_{ij}$   →  aligned"
        head = ("Average gradient cosine similarity: "
                "does the backbone change task-gradient geometry?")
    elif kind == "freq":
        cmap = LinearSegmentedColormap.from_list("freq", SEQUENTIAL)
        vmin, vmax = 0.0, 1.0
        cb_label = r"fraction of batches with cos $\varphi_{ij} < 0$"
        head = "Gradient conflict frequency: which task pairs conflict, and how often?"
    else:
        cmap = LinearSegmentedColormap.from_list("w", SEQUENTIAL)
        vmin, vmax = 0.0, 1.0
        cb_label = "mean $w_{ij}$   0 = leave alone   →   1 = remove fully"
        head = ("AIM intervention weight: does AIM intervene differently "
                "on a pretrained backbone?")

    # columns: heat, heat, its colourbar, [difference, its colourbar]
    pair = len(order) == 2
    widths = [1] * len(order) + [0.05] + ([1, 0.05] if pair else [])
    fig = plt.figure(figsize=(5.4 * (len(order) + (1 if pair else 0)), 5.6))
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths,
                          wspace=0.34, left=0.06, right=0.95,
                          top=0.76, bottom=0.17)

    ims = []
    for s, r in enumerate(order):
        ax = fig.add_subplot(gs[0, s])
        ims.append(_heat(ax, means[r["name"]], tasks, cmap, vmin, vmax,
                         r["name"], annotate=args.annotate,
                         fmt="{:+.2f}" if kind == "cos" else "{:.2f}"))

    cb = fig.colorbar(ims[0], cax=fig.add_subplot(gs[0, len(order)]))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED, labelsize=8)
    # Label on the left, or it lands on the difference panel's row labels.
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(cb_label, color=SECONDARY, fontsize=8.5)

    if pair:
        diff = means[order[0]["name"]] - means[order[1]["name"]]
        dmax = float(np.abs(diff[off]).max()) or 0.05
        ax = fig.add_subplot(gs[0, len(order) + 1])
        dcmap = LinearSegmentedColormap.from_list("conflict", DIVERGING)
        im = _heat(ax, diff, tasks, dcmap, -dmax, dmax,
                   f"{order[0]['name']} − {order[1]['name']}",
                   annotate=args.annotate)
        cb2 = fig.colorbar(im, cax=fig.add_subplot(gs[0, len(order) + 2]))
        cb2.outline.set_visible(False)
        cb2.ax.tick_params(length=0, colors=MUTED, labelsize=8)
        cb2.set_label("difference", color=SECONDARY, fontsize=8.5)

    fig.suptitle(head, x=0.010, y=0.975, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    unit = {"cos": "mean cos", "freq": "conflict rate",
            "weight": "mean w"}[kind]
    stats = "   ·   ".join(
        f"{r['name']}: {unit} over off-diagonal pairs "
        f"{means[r['name']][off].mean():+.3f}"
        for r in order)
    fig.text(0.010, 0.895,
             f"{len(tasks)} QM9 tasks · {args.method.upper()} checkpoint, "
             f"{args.split} split · {len(order[0]['cos'])} batches of "
             f"{args.batch_size} · n_train={args.n_train}, seed {args.seed}",
             fontsize=9.5, color=MUTED, ha="left")
    fig.text(0.010, 0.855, stats, fontsize=9, color=SECONDARY, ha="left")
    fig.text(0.010, 0.030,
             "Diagonal masked (i = j carries no information). Measured here — "
             "train.py never logs the per-step cosines. "
             + ("w_ij combines the measured cosine with tau read from the same "
                "checkpoint; rows are the task being modified."
                if kind == "weight" else
                "LS isolates the backbone: AIM's geometry is partly its own doing."),
             fontsize=7.5, color=MUTED, ha="left")
    return fig


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mean gradient cosine matrix, Uni-Mol vs GNN.")
    ap.add_argument("--backbones", nargs="+", default=["unimol", "gnn"],
                    choices=list(KEY))
    ap.add_argument("--method", default="ls")
    ap.add_argument("--split", default="primary",
                    choices=["primary", "guidance", "val", "test"])
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="task names/indices; default = all 11")
    ap.add_argument("--n_train", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=64)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--data_root", default="data/qm9")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="figures",
                    help="where the figures go")
    ap.add_argument("--cache_dir", default="figures",
                    help="where the measurement cache lives; keep it fixed so "
                         "moving --out never forces a re-measure")
    ap.add_argument("--k", type=float, default=10.0,
                    help="AIM gate temperature for w_ij (Eq. 1); must match "
                         "the k the policy was trained with")
    ap.add_argument("--annotate", action="store_true",
                    help="print the value in every cell (busy at 11 tasks)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    n_tag = len(args.tasks) if args.tasks else 11
    stem = (f"cosine_matrix_{n_tag}task_{args.method}_{args.split}_"
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
            sys.exit("\nNothing measured. Check the checkpoints and that this "
                     "interpreter has the backbone's dependencies.")
        save_csv(cache, results)
        print(f"\nsaved {cache}")

    off = None
    print()
    for r in results:
        n = len(r["tasks"])
        off = ~np.eye(n, dtype=bool)
        M = r["cos"].mean(axis=0)
        print(f"  {r['name']:9s} mean off-diagonal cos {M[off].mean():+.4f}   "
              f"{100 * (r['cos'][:, off] < 0).mean():.1f}% of pair-batches "
              f"conflicting")

    if args.measure_only:
        print("\n--measure-only: nothing plotted.\n"
              f"    python {Path(__file__).name} --plot-only")
        return

    # Three figures come out of one measurement: the cosine matrix itself, the
    # sign of it (conflict frequency), and -- when the checkpoint carries a
    # policy -- Eq. 1's gate. w_ij needs tau, so it only exists for AIM runs.
    kinds = ["cos", "freq"]
    if all(r.get("tau") is not None for r in results):
        kinds.append("weight")
    else:
        print("\n(no tau in this checkpoint -- skipping w_ij; "
              "use --method aim_matrix for the intervention-weight map)")

    names = {"cos": "cosine_similarity", "freq": "conflict_frequency",
             "weight": "aim_weight"}
    comp = out / "comparison"
    comp.mkdir(parents=True, exist_ok=True)

    for kind in kinds:
        fig = make_figure(results, args, kind)
        p = comp / f"{names[kind]}_{stem_tail(args)}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}")

        # Per-backbone copies: the matrix on its own, for the backbone folders.
        order, mats = panels(results, args, kind)
        for r in order:
            d = out / r["name"].replace("-", "")
            d.mkdir(parents=True, exist_ok=True)
            np.savetxt(d / f"{names[kind]}_{stem_tail(args)}.csv",
                       mats[r["name"]], delimiter=",",
                       header=",".join(r["tasks"]), comments="")
        print(f"       per-backbone matrices -> {out}/<backbone>/")


if __name__ == "__main__":
    main()
