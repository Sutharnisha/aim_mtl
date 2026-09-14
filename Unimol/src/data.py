"""
QM9 dataset loading for AIM + Uni-Mol multi-task project.

Provides the full 11-task QM9 property set used by the AIM paper
(Minot & Schneider, NeurIPS AI4Science 2025; arXiv:2509.25955, Sec. 3.2 /
Fig. 3): mu, alpha, eps_HOMO, eps_LUMO, R2, zpve, U0, U, H, G, Cv — the
SAME task set, units, splits and normalization as GNN/src/data.py, so the
two backbones differ only in the encoder. Each molecule is exposed as atomic
numbers + 3-D coordinates (what Uni-Mol consumes); the MPNN node features
of the GNN pipeline are not needed here.

Property units (raw CSV units per the QM9_README, converted to the
standard MPNN-benchmark units — Hartree energies -> eV, matching the
Gilmer 2017 / Nash-MTL / FAMO / PyG QM9 convention):

  mu        Dipole moment                         Debye  (unconverted)
  alpha     Isotropic polarizability               Bohr^3 (unconverted)
  eps_HOMO  HOMO energy                            Hartree -> eV
  eps_LUMO  LUMO energy                            Hartree -> eV
  R2        Electronic spatial extent               Bohr^2 (unconverted)
  zpve      Zero point vibrational energy          Hartree -> eV
  U0        Internal energy at 0K                  Hartree -> eV
  U         Internal energy at 298.15K             Hartree -> eV
  H         Enthalpy at 298.15K                    Hartree -> eV
  G         Free energy at 298.15K                 Hartree -> eV
  Cv        Heat capacity at 298.15K               cal/(mol*K) (unconverted)

Note: the AIM paper's 11-task QM9 subset does NOT include "gap" (HOMO-LUMO
gap is redundant with eps_HOMO/eps_LUMO and is excluded); it DOES include
Cv, which is why the CSV column list below differs from a naive first-11
read of gdb9.sdf.csv.

Training split = 80% primary + 10% guidance + 10% validation
(guidance set used only for AIM policy loss; Section 3.2 of arXiv:2509.25955)

QM9Lazy reads directly from raw gdb9.sdf + gdb9.sdf.csv, indexing molecules
by byte offset via RDKit's SDMolSupplier instead of going through PyG's
InMemoryDataset.process() pipeline. Parsed molecules are cached in memory
on first access (~20-30k molecules for the 10k-train + 10k-test splits, a
few hundred MB at most) so 300-400 epochs don't re-run RDKit parsing on
every epoch — same as the GNN loader.
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from unimol_collate import UniMolCollator, batch_to_device


# ---------------------------------------------------------------------------
# QM9 target definitions
# ---------------------------------------------------------------------------

# Full 11-task QM9 property set used by the AIM paper (Sec. 3.2 / Fig. 3).
# Order matches _CSV_PROP_COLS / _CONVERSIONS below — identical to GNN/src/data.py.
QM9_TASK_NAMES: Dict[int, str] = {
    0:  "mu",       1:  "alpha",  2:  "eps_HOMO", 3: "eps_LUMO",
    4:  "R2",       5:  "zpve",   6:  "U0",       7: "U",
    8:  "H",        9:  "G",      10: "Cv",
}

N_TASKS     = 11
TASK_NAMES  = ["mu", "alpha", "eps_HOMO", "eps_LUMO", "R2", "zpve",
               "U0", "U", "H", "G", "Cv"]
TARGET_COLS = list(range(N_TASKS))

# Per-task units AFTER the conversions below are applied (for reporting).
TASK_UNITS = {
    "mu": "D", "alpha": "Bohr^3", "eps_HOMO": "eV", "eps_LUMO": "eV",
    "R2": "Bohr^2", "zpve": "eV", "U0": "eV", "U": "eV", "H": "eV",
    "G": "eV", "Cv": "cal/(mol*K)",
}

# Hartree → eV conversion (same as PyG QM9)
HAR2EV = 27.211386246

# Hartree -> eV for the 8 energy-valued properties; identity for mu
# (Debye), alpha (Bohr^3), R2 (Bohr^2), Cv (cal/mol*K) — matches the
# PyG QM9 / Gilmer et al. (2017) convention. Order matches _CSV_PROP_COLS.
_CONVERSIONS = np.array([
    1.0, 1.0, HAR2EV, HAR2EV, 1.0, HAR2EV,
    HAR2EV, HAR2EV, HAR2EV, HAR2EV, 1.0,
], dtype=np.float32)

# CSV column names for the 11 properties (in order)
_CSV_PROP_COLS = ["mu", "alpha", "homo", "lumo", "r2",
                  "zpve", "u0", "u298", "h298", "g298", "cv"]


# ---------------------------------------------------------------------------
# Simple data container  (mimics PyG Data interface used by the collator)
# ---------------------------------------------------------------------------

class _MolData:
    """Lightweight molecule container with .z, .pos, .y attributes."""
    __slots__ = ("z", "pos", "y")

    def __init__(self, z, pos, y):
        self.z   = z    # LongTensor  [n_atoms]
        self.pos = pos  # FloatTensor [n_atoms, 3]
        self.y   = y    # FloatTensor [1, 11]


# ---------------------------------------------------------------------------
# QM9 dataset  (reads raw SDF on demand via RDKit)
# ---------------------------------------------------------------------------

class QM9Lazy(Dataset):
    """
    QM9 dataset backed by the raw SDF/CSV files.

    - Properties are pre-loaded from gdb9.sdf.csv (27 MB).
    - Atom types and coordinates are read on demand from gdb9.sdf using
      RDKit SDMolSupplier, which builds a byte-offset index rather than
      loading all molecules at once.
    - Uncharacterized molecules (3054 total) are filtered out as in PyG.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self._cache: Dict[int, "_MolData"] = {}
        raw_dir   = self.root / "raw"

        # ── Load property CSV ──────────────────────────────────────────────
        csv_path = raw_dir / "gdb9.sdf.csv"
        df = pd.read_csv(csv_path, usecols=["mol_id"] + _CSV_PROP_COLS)
        raw_props = df[_CSV_PROP_COLS].values.astype(np.float32)
        self._targets = torch.from_numpy(raw_props * _CONVERSIONS)  # [N_total, 11]

        # ── Parse excluded molecule indices (1-based in file → 0-based) ───
        excluded = set()
        unchar_path = raw_dir / "uncharacterized.txt"
        if unchar_path.exists():
            with open(unchar_path) as f:
                for line in f:
                    m = re.match(r"^\s*(\d+)\s", line)
                    if m:
                        excluded.add(int(m.group(1)) - 1)  # 1-based → 0-based

        # ── Build valid index list ─────────────────────────────────────────
        n_total = len(self._targets)
        self._valid_idx = [i for i in range(n_total) if i not in excluded]

        # ── RDKit lazy SDF supplier ────────────────────────────────────────
        try:
            from rdkit import Chem
            sdf_path = str(raw_dir / "gdb9.sdf")
            self._supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        except ImportError:
            raise ImportError("RDKit is required. Install with: conda install -c conda-forge rdkit")

    def __len__(self) -> int:
        return len(self._valid_idx)

    def __getitem__(self, idx: int) -> _MolData:
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        mol_idx = self._valid_idx[idx]
        mol     = self._supplier[mol_idx]

        if mol is None:
            raise ValueError(f"RDKit returned None for SDF index {mol_idx}. SDF may be corrupted.")

        # Atomic numbers and 3-D coordinates from first conformer
        conf = mol.GetConformer()
        z    = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.long,
        )
        pos  = torch.tensor(conf.GetPositions(), dtype=torch.float32)  # [n, 3]
        y    = self._targets[mol_idx].unsqueeze(0)                     # [1, 11]

        data = _MolData(z=z, pos=pos, y=y)
        self._cache[idx] = data
        return data


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_qm9(root: str = "../../data/qm9") -> QM9Lazy:
    """Return a memory-efficient QM9 dataset backed by raw SDF/CSV files."""
    root_path = Path(root)
    sdf_path  = root_path / "raw" / "gdb9.sdf"
    csv_path  = root_path / "raw" / "gdb9.sdf.csv"

    if not sdf_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"QM9 raw files not found in {root_path / 'raw'}.\n"
            "Run download_data.py first, or download manually from:\n"
            "  https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904"
        )

    return QM9Lazy(root)


def make_splits(
    dataset,
    n_train:    int   = 10_000,
    val_frac:   float = 0.10,
    guide_frac: float = 0.10,
    seed:       int   = 42,
) -> Dict[str, Subset]:
    """
    Partition QM9 following AIM paper Section 3.2:
      - n_train total training molecules
      - guide_frac × n_train  →  guidance set (policy loss only)
      - val_frac   × n_train  →  validation set
      - remainder             →  primary training set
      - up to 10k additional  →  held-out test set
    """
    rng = np.random.default_rng(seed)
    N   = len(dataset)

    perm    = rng.permutation(N)
    n_guide = int(n_train * guide_frac)
    n_val   = int(n_train * val_frac)
    n_prim  = n_train - n_guide - n_val
    n_test  = min(10_000, N - n_train)

    assert n_prim > 0, f"n_train={n_train} is too small for the requested split fractions."
    assert n_train + n_test <= N, f"Not enough molecules ({N}) for n_train={n_train}."

    idx_prim  = perm[:n_prim].tolist()
    idx_guide = perm[n_prim: n_prim + n_guide].tolist()
    idx_val   = perm[n_prim + n_guide: n_train].tolist()
    idx_test  = perm[n_train: n_train + n_test].tolist()

    return {
        "primary":  Subset(dataset, idx_prim),
        "guidance": Subset(dataset, idx_guide),
        "val":      Subset(dataset, idx_val),
        "test":     Subset(dataset, idx_test),
    }


def compute_normalization(
    subset: Subset,
    target_cols: List[int] = TARGET_COLS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-property mean and std over the primary training split."""
    all_y = torch.cat(
        [data.y[:, target_cols] for data in subset], dim=0
    )   # [N, 11]
    means = all_y.mean(dim=0)
    stds  = all_y.std(dim=0).clamp(min=1e-8)
    return means, stds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_loaders(
    root:        str   = "../../data/qm9",
    n_train:     int   = 10_000,
    batch_size:  int   = 32,
    seed:        int   = 42,
    num_workers: int   = 0,
) -> dict:
    """
    Build DataLoaders for all splits using UniMolCollator.

    Every batch produced by these loaders is a dict with keys:
        src_tokens, src_coord, src_distance, src_edge_type  — Uni-Mol inputs
        targets       : FloatTensor [B, 11]  (physical units, un-normalised)
        n_atoms       : LongTensor  [B]

    Returns dict with keys:
        primary_loader, guide_loader, val_loader, test_loader,
        means [11], stds [11], split_sizes {primary, guidance, val, test}
    """
    dataset = load_qm9(root)
    splits  = make_splits(dataset, n_train=n_train, seed=seed)

    collator = UniMolCollator(target_cols=TARGET_COLS)

    loader_kw = dict(
        collate_fn=collator,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    primary_loader = DataLoader(splits["primary"],  shuffle=True,  **loader_kw)
    guide_loader   = DataLoader(splits["guidance"], shuffle=True,  **loader_kw)
    val_loader     = DataLoader(splits["val"],      shuffle=False, **loader_kw)
    test_loader    = DataLoader(splits["test"],     shuffle=False, **loader_kw)

    means, stds = compute_normalization(splits["primary"])

    return {
        "primary_loader": primary_loader,
        "guide_loader":   guide_loader,
        "val_loader":     val_loader,
        "test_loader":    test_loader,
        "means":          means,   # [11]
        "stds":           stds,    # [11]
        "split_sizes":    {k: len(v) for k, v in splits.items()},
    }


# ---------------------------------------------------------------------------
# Quick inspection helper
# ---------------------------------------------------------------------------

def inspect_qm9(root: str = "../../data/qm9"):
    dataset = load_qm9(root)
    sample  = dataset[0]
    print(f"QM9 size       : {len(dataset):,} molecules")
    print(f"data.z  shape  : {sample.z.shape}   (atomic numbers)")
    print(f"data.pos shape : {sample.pos.shape}  (3-D coordinates)")
    print(f"data.y  shape  : {sample.y.shape}  (all 11 properties — collator slices TARGET_COLS)")
    print(f"We use cols    : {TARGET_COLS}  ({N_TASKS} tasks: {TASK_NAMES})")
    print(f"Task names     : {TASK_NAMES}")
    return dataset


if __name__ == "__main__":
    inspect_qm9()
    loaders = get_loaders(n_train=10_000)
    print("\nSplit sizes:", loaders["split_sizes"])
    # Verify one batch shape
    batch = next(iter(loaders["primary_loader"]))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:16s}: {v.shape}")
