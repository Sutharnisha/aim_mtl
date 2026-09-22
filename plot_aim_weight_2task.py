"""
AIM intervention weight w_ij -- Uni-Mol vs GNN, 2-task setting.

Question: does AIM intervene differently on a pretrained encoder than on one
trained from scratch?

    Eq. 1   w_ij = sigma( k * (tau_ij - cos(g_i, g_j)) )

w_ij in (0,1) is how much of task j's direction AIM removes from task i's
gradient: 0 = leave it alone, 1 = remove the whole projection. It has two
inputs, and they come from different places:

    tau_ij   learned, read from the AIM checkpoint's policy state
    cos ij   MEASURED here -- never logged by train.py

Unlike the cosine, w is NOT symmetric: AIM-Matrix learns one tau per ORDERED
pair, so w(mu <- eps_LUMO) and w(eps_LUMO <- mu) differ. With two tasks that
gives two meaningful cells per backbone, which is why a heatmap earns its place
here where it did not for the raw cosine.

Measured at the AIM checkpoint (not LS): tau and cos have to come from the same
model state, or w is a quantity that never existed during training.

Two phases, because matplotlib crashes inside the conda env that has
unimol_tools:

    # measure (env with torch + rdkit + unimol_tools)
    <aim_gnn python> plot_aim_weight_2task.py --measure-only
    # plot (env with a working matplotlib)
    python plot_aim_weight_2task.py --plot-only

Saves into figures/:
    aim_weight_2task_<method>_<split>_n<N>_seed<S>.png
    reuses the cosine cache written by plot_cosine_2task.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_cosine_2task import (  # noqa: E402
    KEY, SERIES, SURFACE, INK, SECONDARY, MUTED, AXIS,
    measure, save_csv, load_csv,
)

# Sequential blue ramp (magnitude, one hue light->dark) for w in [0, 1].
SEQ = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]


def _style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.labelsize": 9.5, "ytick.labelsize": 10,
        "legend.frameon": False, "figure.dpi": 140,
    })


def weights(r: dict, k: float) -> np.ndarray:
    """w[b, i, j] for every batch b -- sigma(k (tau_ij - cos_b))."""
    tau = r["tau"]
    cos = r["cos"]                                  # symmetric, one per batch
    return 1.0 / (1.0 + np.exp(-k * (tau[None, :, :] - cos[:, None, None])))


def make_figure(results: List[dict], args) -> plt.Figure:
    _style()
    order = [r for n in SERIES for r in results if r["name"] == n]
    cmap = LinearSegmentedColormap.from_list("w", SEQ)
    tasks = order[0]["tasks"]

    fig = plt.figure(figsize=(12.6, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.06, 1.6], wspace=0.40,
                          left=0.06, right=0.96, top=0.68, bottom=0.17)

    ims = []
    for s, r in enumerate(order):
        ax = fig.add_subplot(gs[0, s])
        W = weights(r, args.k).mean(axis=0)
        shown = W.copy()
        np.fill_diagonal(shown, np.nan)             # j != i; diagonal unused
        cmap.set_bad(SURFACE)
        im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=1.0)
        ims.append(im)
        ax.set_xticks(range(2), tasks)
        ax.set_yticks(range(2), tasks)
        ax.set_xlabel("removes the direction of  j", fontsize=8.5, color=MUTED)
        ax.set_ylabel("gradient of  i", fontsize=8.5, color=MUTED)
        ax.set_title(r["name"], color=INK, loc="left", pad=8, fontsize=11)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        for i in range(2):
            for j in range(2):
                if i == j:
                    continue
                v = W[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13,
                        color=SURFACE if v > 0.55 else INK)
                ax.text(j, i + 0.22, f"τ {r['tau'][i, j]:+.3f}",
                        ha="center", va="center", fontsize=8,
                        color=SURFACE if v > 0.55 else SECONDARY)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(ims[-1], cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED, labelsize=8)
    cb.set_label("mean $w_{ij}$", color=SECONDARY, fontsize=8.5)

    # Per-batch spread: four numbers averaged over batches hide their variance.
    ax = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(0)
    rows, ticks = [], []
    for r in order:
        W = weights(r, args.k)
        for (i, j) in ((0, 1), (1, 0)):
            rows.append((r["name"], W[:, i, j]))
            ticks.append(f"{tasks[i]} ← {tasks[j]}")
    for y, ((name, v), lab) in enumerate(zip(rows, ticks)):
        yy = len(rows) - 1 - y
        c = SERIES[name]
        ax.scatter(v, np.full(v.size, yy) + (rng.random(v.size) - 0.5) * 0.22,
                   s=22, color=c, alpha=0.40, linewidths=0, zorder=3)
        ax.scatter([v.mean()], [yy], s=110, color=c, edgecolor=SURFACE, lw=2,
                   zorder=5)
    ax.set_yticks(range(len(rows)),
                  [f"{n}   {t}" for (n, _), t in zip(rows, ticks)][::-1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axvline(0.5, color=AXIS, lw=1.0, zorder=2)
    ax.text(0.5, 1.01, "τ = cos φ", color=MUTED, fontsize=8,
            ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel("$w_{ij}$  per batch")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Spread across batches", color=INK, loc="left", pad=8,
                 fontsize=11)

    fig.suptitle("Does AIM intervene differently on a pretrained backbone?",
                 x=0.010, y=0.975, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.010, 0.875,
             f"Eq. 1  $w_{{ij}} = \\sigma(k(\\tau_{{ij}} - \\cos\\varphi_{{ij}}))$, "
             f"k = {args.k:g}  ·  {args.method} checkpoint, {args.split} split, "
             f"{len(order[0]['cos'])} batches  ·  n_train={args.n_train}, "
             f"seed {args.seed}",
             fontsize=9.5, color=MUTED, ha="left")
    fig.text(0.010, 0.035,
             "$\\tau$ is read from the checkpoint's policy state; "
             "$\\cos\\varphi$ is measured here (train.py never logs it). "
             "Rows are the task being modified, columns the task doing the "
             "modifying — w is not symmetric.",
             fontsize=7.5, color=MUTED, ha="left")
    return fig


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="AIM intervention weight w_ij, 2-task, Uni-Mol vs GNN.")
    ap.add_argument("--backbones", nargs="+", default=["unimol", "gnn"],
                    choices=list(KEY))
    ap.add_argument("--method", default="aim_matrix",
                    choices=["aim_matrix", "aim_scalar"],
                    help="only AIM runs carry a policy / tau")
    ap.add_argument("--split", default="primary",
                    choices=["primary", "guidance", "val", "test"])
    ap.add_argument("--tasks", nargs="+", default=["mu", "eps_LUMO"])
    ap.add_argument("--k", type=float, default=10.0, help="gate temperature")
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stem = (f"aim_weight_2task_{args.method}_{args.split}_"
            f"n{args.n_train}_seed{args.seed}")
    # Same cache file the cosine script writes, so the measurement is shared.
    cdir = ROOT / args.cache_dir
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / (f"cosine_2task_{args.method}_{args.split}_"
                    f"n{args.n_train}_seed{args.seed}.csv")

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
            sys.exit("\nNothing measured. Check the AIM checkpoints and that this "
                     "interpreter has the backbone's dependencies.")
        save_csv(cache, results)
        print(f"\nsaved {cache}")

    missing = [r["name"] for r in results if r.get("tau") is None]
    if missing:
        sys.exit(f"\nNo tau for {', '.join(missing)} -- the {args.method} "
                 f"checkpoint has no policy state. w_ij cannot be computed.")

    print()
    for r in results:
        W = weights(r, args.k)
        m = W.mean(axis=0)
        print(f"  {r['name']:9s} cos {r['cos'].mean():+.3f}   "
              f"w[{r['tasks'][0]}<-{r['tasks'][1]}] {m[0,1]:.3f}   "
              f"w[{r['tasks'][1]}<-{r['tasks'][0]}] {m[1,0]:.3f}")

    if args.measure_only:
        print("\n--measure-only: nothing plotted.\n"
              f"    python {Path(__file__).name} --plot-only")
        return

    fig = make_figure(results, args)
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    print(f"\nsaved {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
