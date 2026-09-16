# AIM: Multi-Task Molecular Property Prediction

Two implementations of **AIM** (Adaptive Intervention for Multi-task Learning) for molecular property prediction on QM9, sharing the same algorithm, training protocol, and evaluation code, but with different encoder backbones:

- **[`GNN/`](GNN/)** — an MPNN (Gilmer et al., 2017) trained from scratch.
- **[`Unimol/`](Unimol/)** — a pretrained Uni-Mol SE(3)-Transformer, fine-tuned.

> **Paper:** Minot & Schneider — *AIM: Adaptive Intervention for Deep Multi-task Learning of Molecular Properties*, NeurIPS AI4Science 2025. [arXiv:2509.25955](https://arxiv.org/abs/2509.25955)

The paper itself only ever benchmarks one GNN backbone; testing AIM with a second, unrelated architecture (Uni-Mol) is this repo's own extension of the paper's suggested future work — *"evaluating AIM's generalizability across diverse model architectures beyond the GNN used here."* To make that a fair test, every hyperparameter that isn't intrinsically tied to a specific encoder (epochs, batch size, learning rates, policy temperature, loss weights, task-head width, etc.) is kept **identical** between the two pipelines — see [Training](#training) below. Only the encoder itself differs.

---

## What AIM does

Training one encoder on several molecular property targets at once runs into gradient conflicts: tasks pull the shared parameters in different directions. Static heuristics like PCGrad resolve this with a fixed geometric rule. AIM instead **learns** a per-task-pair policy — a threshold τ per task pair — that decides how much of each task's conflicting gradient component to project out, trained jointly with the main model using a guidance loss plus two differentiable regularizers (magnitude preservation, progress-on-hard-tasks). `GNN/` compares AIM (scalar and matrix policy variants) against linear scalarization (LS), PCGrad, and single-task learning (STL) baselines on the AIM paper's **full 11-task QM9 property set** (mu, alpha, eps_HOMO, eps_LUMO, R2, zpve, U0, U, H, G, Cv — see `GNN/src/data.py` for units) at the paper's **n_train=10,000** setting. (The paper's own comparison also includes CAGrad/Nash-MTL/FAMO; this repo intentionally sticks to LS/PCGrad/AIM/STL only.) `Unimol/` runs the **same** 11-task, n_train=10,000 protocol with the same methods; the two pipelines mirror each other file-for-file except for the encoder (see [Training](#training)).

Both pipelines assume a normal single-GPU machine — there is no reduced-memory or CPU-offload mode; per-task gradients are computed from one shared forward pass via `torch.autograd.grad` and combined directly on-device.

---

## Repository layout

```
├── GNN/
│   ├── src/                 # training pipeline (see below)
│   └── result_updated/        # trained runs: <method>_n<N>_seed<S>/{best_model.pt, history.json}
│                              #   (--tasks subsets: tasks_<names>/<method>_n<N>_seed<S>/)
├── Unimol/
│   ├── src/                 # training pipeline (see below)
│   └── results/             # trained runs: <method>_n<N>_seed<S>/{best_model.pt, history.json}
└── data/qm9/raw/             # shared QM9 data (gdb9.sdf, gdb9.sdf.csv, uncharacterized.txt)
```

`GNN/src/` and `Unimol/src/` mirror each other file-for-file wherever the logic isn't backbone-specific:

| File | Role | Backbone-specific? |
|---|---|---|
| `train.py` | Training loop + CLI entry point | Yes — different `build_model` call, otherwise identical structure |
| `model.py` | Encoder + multi-task heads | Yes — MPNN vs. Uni-Mol |
| `data.py` | QM9 loading, splits, normalization | Path defaults only |
| `gnn_collate.py` / `unimol_collate.py` | Batch collation (padded tensors) | Yes — different input formats |
| `aim_optimizer.py` | AIM policy + gradient intervention (Eq. 1–6) | **No — byte-identical in both** |
| `baselines.py` | LS / PCGrad gradient combination | **No — byte-identical in both** |
| `metrics.py` | Mean Rank, Δm% | **No — byte-identical in both** |
| `analysis.py` | Results table + policy-matrix/loss plots for that backbone alone | No (same pattern, GNN's is the maintained one — see note below) |


Both `src/` folders have `run_10k_experiments.py` (full STL + MTL method × seed matrix, 11-task or any `--tasks` subset); `GNN/src/` also has `run_10k_stl.py` (STL baseline only). Uni-Mol's `src/` additionally has `download_data.py` (QM9 + pretrained-weight setup/smoke test), `environment.yml`, and the cross-backbone comparison tools `comparison_table.py` / `plot_comparison.py` (these read both `Unimol/results/` and `GNN/result_updated/` to compare methods across both backbones side by side).

> **Note:** `Unimol/src/analysis.py` is a stale leftover from before this repo was split into `GNN/`+`Unimol/` — it's superseded by `comparison_table.py`/`plot_comparison.py`, has unused imports, and its `plot_loss_curves`  assumes the project 3-task setup. It isn't part of the intended pipeline; flagging it here rather than silently documenting it as a real feature.

---

## Installation

One virtual environment at the repo root serves **both** pipelines; there is a single `requirements.txt`:

```bash
python -m venv .venv                     # Python 3.11+ (tested on 3.14)
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # match cuXX to your driver; drop --index-url for CPU
pip install -r requirements.txt
```

`rdkit` (QM9 parsing), `unimol-tools` (pretrained Uni-Mol backbone) and the plotting stack are all pinned there. Pretrained Uni-Mol weights (~700 MB) and the atom dictionary download automatically on first use into the package's `weights/` folder.

---

## Data

Both pipelines read from the same location. Download QM9 raw files and place them at:

```
data/qm9/raw/gdb9.sdf
data/qm9/raw/gdb9.sdf.csv
data/qm9/raw/uncharacterized.txt   # optional exclusion list
```

QM9 is available from [quantum-machine.org/datasets](http://quantum-machine.org/datasets/) or via PyTorch Geometric's `torch_geometric.datasets.QM9`. `python download_data.py` (from `Unimol/src/`) downloads QM9, triggers the pretrained-weight download, and runs a quick Uni-Mol pipeline smoke test.

---

## Training

> **GNN and Uni-Mol run the same protocol.** Both pipelines use the AIM
> paper's QM9 setting: the **full 11-task property set** (mu, alpha,
> eps_HOMO, eps_LUMO, R2, zpve, U0, U, H, G, Cv — see the unit table in
> `data.py`) at **n_train=10,000**, 400 epochs with early stopping
> (patience 75), batch size 32, ReduceLROnPlateau (patience 50) on the
> **normalized** validation MAE, task heads 128/512 → 64 → 32 → 1 with ReLU,
> and the same LS / PCGrad / AIM / STL code. Methods are LS, PCGrad, AIM
> (scalar/matrix), and STL — the paper's comparison also includes
> CAGrad/Nash-MTL/FAMO, intentionally not added here.
>
> The only intentional differences are tied to the backbone itself:
> `Unimol/` fine-tunes a **pretrained** Uni-Mol transformer, so its default
> `--lr_model` is `1e-4` (the paper's `{1e-3, 5e-4}` grid is for a
> from-scratch GNN and is unstable for fine-tuning) and it exposes
> `--freeze_backbone` / `--trainable_layers N` as ablations (default `-1`,
> full fine-tune, matching the GNN where every layer trains). The Uni-Mol
> collator builds inputs exactly as `unimol_tools` does for the pretrained
> weights: `[CLS] + atoms + [SEP]`, molecule-centred coordinates, and
> edge types keyed by the full pretrained dictionary length.
>
> **Model selection:** MTL runs select/early-stop on the mean normalized
> val MAE; **STL runs select on the trained task only** (the other ten
> heads are untrained noise and must not drive checkpointing). Every
> `history.json` records the metric used as `val_select`, and all table /
> plot scripts pick the best epoch with `metrics.best_epoch_key`, so the
> reported test MAE is always the one at `best_model.pt`.

### Running commands

All commands are run from `GNN/src/` (MPNN backbone) or `Unimol/src/`
(Uni-Mol backbone) — the CLIs are identical. Use the project venv
(`.venv/bin/python`), which has torch installed.

**Task selection.** Every script takes `--tasks` (names or indices).
Omit it for the paper's full 11-task setting; pass a subset for a smaller
problem. Subset runs are stored in their own sub-folder
(`<save_dir>/tasks_<names>/`, e.g. `tasks_mu-eps_LUMO/`) so they never
overwrite 11-task runs, and `--resume` works independently in each.

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| name | mu | alpha | eps_HOMO | eps_LUMO | R2 | zpve | U0 | U | H | G | Cv |

#### 1. Full method × seed matrix — one command (recommended)

`run_10k_experiments.py` runs STL (one run per task) plus every MTL method
(`ls`, `pcgrad`, `aim_scalar`, `aim_matrix`) at n_train=10,000 and prints the
paper-style table (Mean Rank, Δm% vs STL, per-task test MAE).

```bash
# ── 11 tasks (paper setting) ──────────────────────────────────────────
python run_10k_experiments.py                                # 1 seed (42)
python run_10k_experiments.py --seeds 42 43 44                # 3 seeds: mean ± std + significance vs STL
python run_10k_experiments.py --resume                        # skip runs that already have history.json
python run_10k_experiments.py --methods aim_matrix --skip_stl # only one MTL method, no STL
python run_10k_experiments.py --n_epochs 5 --patience 5       # smoke test

# ── 2 tasks (mu + eps_LUMO) — same script, same methods ───────────────
python run_10k_experiments.py --tasks mu eps_LUMO
python run_10k_experiments.py --tasks mu eps_LUMO --seeds 42 43 44
python run_10k_experiments.py --tasks mu eps_LUMO --resume
python run_10k_experiments.py --tasks 0 3                     # indices work too
python run_10k_experiments.py --tasks mu eps_LUMO --methods ls pcgrad   # subset of methods
python run_10k_experiments.py --tasks mu eps_LUMO --skip_stl  # MTL only, no STL baseline
```

Results: `<save_dir>/seed_results_n10000.json` (11-task) or
`<save_dir>/tasks_mu-eps_LUMO/seed_results_n10000.json` (2-task), with
`save_dir = ../result_updated` (GNN) or `../results` (Uni-Mol).

#### 2. STL baseline only

```bash
python run_10k_stl.py                        # GNN only: all 11 STL runs, seed 42
python run_10k_stl.py --seeds 42 43 44
python run_10k_stl.py --tasks mu eps_LUMO    # just these STL runs (11-head model, only one head trained)
python run_10k_stl.py --resume
```

For a 2-task STL baseline that lives alongside the 2-task MTL runs, prefer
the main runner: `python run_10k_experiments.py --tasks mu eps_LUMO --methods`
(an empty `--methods` list runs STL only).

#### 3. Single runs with `train.py`

```bash
# ── 11 tasks ──────────────────────────────────────────────────────────
python train.py --method ls         --n_train 10000 --n_epochs 400   # linear scalarization
python train.py --method pcgrad     --n_train 10000 --n_epochs 400   # PCGrad
python train.py --method aim_scalar --n_train 10000 --n_epochs 400   # AIM, scalar policy
python train.py --method aim_matrix --n_train 10000 --n_epochs 400   # AIM, matrix policy
python train.py --method stl --stl_task_idx 0 --n_train 10000 --n_epochs 400   # STL on mu
python train.py --method stl --stl_task_idx 3 --n_train 10000 --n_epochs 400   # STL on eps_LUMO
for i in $(seq 0 10); do python train.py --method stl --stl_task_idx $i; done  # all 11 STL runs

# ── 2 tasks (mu + eps_LUMO): add --tasks; --stl_task_idx keeps its global index
python train.py --method ls         --tasks mu eps_LUMO
python train.py --method pcgrad     --tasks mu eps_LUMO
python train.py --method aim_scalar --tasks mu eps_LUMO
python train.py --method aim_matrix --tasks mu eps_LUMO
python train.py --method stl --stl_task_idx 0 --tasks mu eps_LUMO   # STL on mu
python train.py --method stl --stl_task_idx 3 --tasks mu eps_LUMO   # STL on eps_LUMO

# Other subsets work the same way, e.g. the three energies:
python train.py --method aim_matrix --tasks U0 U H
```

Note: `train.py` on its own does **not** add the `tasks_<names>/` sub-folder —
pass `--save_dir ../result_updated/tasks_mu-eps_LUMO` yourself if you want a
2-task single run to sit next to the runner's output.

#### 4. Uni-Mol specifics

Same commands from `Unimol/src/`, plus the fine-tuning ablations:

```bash
python run_10k_experiments.py                                   # 11 tasks, ls/pcgrad/aim_scalar/aim_matrix + STL
python run_10k_experiments.py --tasks mu eps_LUMO               # 2 tasks
python train.py --method aim_matrix --tasks mu eps_LUMO --trainable_layers 4   # partial fine-tune
python train.py --method aim_matrix --freeze_backbone           # heads only
```

**Key GNN arguments (paper-matched defaults):**

| Argument | Default | Description |
|---|---|---|
| `--method` | `aim_matrix` | `ls` / `pcgrad` / `aim_scalar` / `aim_matrix` / `stl` |
| `--tasks` | all 11 | Task names or indices to train on, e.g. `--tasks mu eps_LUMO` (model gets one head per selected task) |
| `--stl_task_idx` | `0` | STL only: global QM9 task index 0..10 (must be among `--tasks` when a subset is used) |
| `--n_train` | `10000` | AIM paper QM9 subset size (paper also reports 50k/100k) |
| `--n_epochs` | `400` | Paper range: 300-400 |
| `--patience` | `75` | Early-stopping patience on normalized val MAE (paper: 75) |
| `--batch_size` | `32` | Batch size |
| `--head_hidden` | `64` | Task-head hidden width |
| `--trainable_layers` | `-1` | `-1` = all layers trainable (GNN has no pretrained weights to freeze) |
| `--lr_model` | `5e-4` | Encoder + head learning rate (paper grid: `{1e-3, 5e-4}`) |
| `--lr_policy` | `1e-3` | AIM policy learning rate (paper grid: `{5e-4, 1e-3, 5e-3}`) |
| `--lambda_g` | `1.0` | Policy loss — guidance weight (paper: `[0, 1]`) |
| `--lambda_m` | `0.01` | Policy loss — magnitude weight (paper: `[0, 0.01]`) |
| `--lambda_p` | `0.08` | Policy loss — progress weight (paper: `{0, 0.08, 0.8}`) |
| `--k` | `10.0` | Policy sigmoid temperature |
| `--data_root` | `../../data/qm9` | Directory depth from `GNN/src/` |
| `--save_dir` | `../result_updated` | Result tree |

Checkpoints (`best_model.pt`) and full per-epoch history (`history.json`) are saved to `<save_dir>/<run_name>/`. Model selection, LR scheduling, and early stopping all use **normalized** (unit-agnostic) validation MAE — with 11 tasks spanning Debye/eV/Bohr²/Bohr³/cal·mol⁻¹K⁻¹ units, averaging raw MAE would let whichever property has the largest numeric scale dominate; `history.json` still records raw physical-unit MAE per task (`val_per_task`/`test_per_task`) for reporting.

> **STL caveat:** `--method stl` only trains the single head named by `--stl_task_idx` — the other heads in that run's `history.json` were never trained and their MAE columns are meaningless noise. To get every property's STL baseline, run it once per task (`--stl_task_idx 0..10`, or each selected task with `--tasks`), or just use `run_10k_experiments.py`, which does this automatically. The MTL methods (`ls`/`pcgrad`/`aim_scalar`/`aim_matrix`) train all selected tasks together in a single run.

To launch the full method × seed matrix at 10k, use `run_10k_experiments.py` from either `GNN/src/` or `Unimol/src/` — the two scripts are identical apart from the default result directory (`GNN/result_updated/` vs `Unimol/results/`); both run `ls`, `pcgrad`, `aim_scalar`, `aim_matrix` and STL, on the 11-task set by default or on any `--tasks` subset. For Uni-Mol the same `train.py` flags apply, plus `--freeze_backbone` / `--trainable_layers`.

---

## Analysis & plotting

**Per-backbone** (run from that backbone's `src/`):
```bash
python analysis.py --results_dir ../result_updated   # GNN: results table + policy heatmap + loss curves
python plot_stl_result.py                            # either backbone: plots every stl_task*_... run found
```

**Cross-backbone** (Uni-Mol's `src/` only, reads both `Unimol/results/` and `../../GNN/result_updated/`; all 11 tasks, `--n_train`/`--seed` select the run set, default 10000 / 42; missing runs are skipped):
```bash
python comparison_table.py           # Mean Rank + Δm% (vs STL / LS / PCGrad) per backbone (terminal + PNG + CSV)
python plot_comparison.py            # per-task val-MAE curves (both backbones), Δm% bars, 11x11 τ heatmaps
python plot_aim_matrix_unimol.py     # AIM-Matrix diagnostics for one Uni-Mol run (--run_name)
```

Both `analysis.py` and `comparison_table.py`/`plot_comparison.py` use the same Δm% convention (positive = better than STL) and the same Mean Rank definition, from the shared `metrics.py`.

---

## Learning resources

- **[`Unimol/src/explanation_train.md`](Unimol/src/explanation_train.md)** — a beginner-oriented, section-by-section walkthrough of `train.py`: reading order, what each function does, the `torch.autograd.grad`/`retain_graph` mechanics behind per-task gradient collection, a worked example trace, and a glossary. Written for an earlier 3-task version of the Uni-Mol `train.py`; the structure is unchanged, but the current file trains 11 tasks and mirrors `GNN/src/train.py`.

---

## Results structure

```
GNN/result_updated/<method>_n<N>_seed<S>/
├── best_model.pt      # checkpoint at the best validation-MAE epoch
└── history.json        # per-epoch: val/test MAE per task, train loss, AIM policy losses + τ (if applicable)

Unimol/results/<method>_n<N>_seed<S>/
├── best_model.pt
└── history.json
```

`<method>` is one of `ls`, `pcgrad`, `aim_scalar`, `aim_matrix`, or `stl_task<i>_<name>`. `<N>` is `--n_train`, `<S>` is `--seed`.

Runs launched with `--tasks` (e.g. the 2-task mu + eps_LUMO problem) use the same layout one level down, so they never collide with 11-task runs:

```
GNN/result_updated/tasks_mu-eps_LUMO/<method>_n<N>_seed<S>/{best_model.pt, history.json}
GNN/result_updated/tasks_mu-eps_LUMO/seed_results_n<N>.json
Unimol/results/tasks_mu-eps_LUMO/...
```

---

## Citation

```bibtex
@inproceedings{minot2025aim,
  title     = {AIM: Adaptive Intervention for Deep Multi-task Learning of Molecular Properties},
  author    = {Minot, Jack and Schneider, Nadine},
  booktitle = {NeurIPS Workshop on AI for Science},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.25955}
}
```
