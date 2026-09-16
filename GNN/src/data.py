"""
QM9 dataset loading for AIM + MPNN (Gilmer et al. 2017) project.

Provides the full 11-task QM9 property set used by the AIM paper
(Minot & Schneider, NeurIPS AI4Science 2025; arXiv:2509.25955, Sec. 3.2 /
Fig. 3): eps_HOMO, eps_LUMO, R2, zpve, U0, U, H, G, Cv, mu, alpha.
Reproducible splits and per-task normalization are provided; each molecule
is represented by an 11-dim atom feature vector per atom (matching the PyG
QM9.x format used across MPNN benchmarks on QM9).

Property units (raw CSV units per the QM9_README, converted to the
standard MPNN-benchmark units — Hartree energies -> eV, matching the
Gilmer 2017 / Nash-MTL / FAMO / PyG QM9 convention that the AIM paper's
GNN backbone descends from):

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

Sanitization note:
  QM9 SDF is loaded with sanitize=False for compatibility, then each
  molecule is sanitized on-demand inside __getitem__ using
  Chem.SanitizeMol(mol, catchErrors=True) so hybridization / aromaticity
  / numH fields are correctly populated.  All 130k+ QM9 molecules pass.
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from gnn_collate import GNNCollator, batch_to_device


# ---------------------------------------------------------------------------
# QM9 target definitions
# ---------------------------------------------------------------------------

# Full 11-task QM9 property set used by the AIM paper (Sec. 3.2 / Fig. 3).
# Order matches _CSV_PROP_COLS / _CONVERSIONS below.
QM9_TASK_NAMES: Dict[int, str] = {
    0:  "mu",       1:  "alpha",  2:  "eps_HOMO", 3: "eps_LUMO",
    4:  "R2",       5:  "zpve",   6:  "U0",       7: "U",
    8:  "H",        9:  "G",      10: "Cv",
}

N_TASKS     = 11
TASK_NAMES  = ["mu", "alpha", "eps_HOMO", "eps_LUMO", "R2", "zpve",
               "U0", "U", "H", "G", "Cv"]
TARGET_COLS = list(range(N_TASKS))


def parse_task_spec(tasks) -> List[int]:
    """Turn a list of task names ('mu') and/or indices ('3') into sorted
    QM9 column indices. None / empty -> all N_TASKS tasks."""
    if not tasks:
        return list(TARGET_COLS)
    idxs = []
    for tk in tasks:
        tk = str(tk)
        if tk.isdigit():
            i = int(tk)
            if not 0 <= i < N_TASKS:
                raise ValueError(f"task index {i} out of range 0..{N_TASKS - 1}")
        elif tk in TASK_NAMES:
            i = TASK_NAMES.index(tk)
        else:
            raise ValueError(f"unknown task {tk!r}; choose from {TASK_NAMES}")
        idxs.append(i)
    return sorted(set(idxs))

# Per-task units AFTER the conversions below are applied (for reporting).
TASK_UNITS = {
    "mu": "D", "alpha": "Bohr^3", "eps_HOMO": "eV", "eps_LUMO": "eV",
    "R2": "Bohr^2", "zpve": "eV", "U0": "eV", "U": "eV", "H": "eV",
    "G": "eV", "Cv": "cal/(mol*K)",
}

HAR2EV = 27.211386246

# Hartree -> eV for the 8 energy-valued properties; identity for mu
# (Debye), alpha (Bohr^3), R2 (Bohr^2), Cv (cal/mol*K) — matches the
# PyG QM9 / Gilmer et al. (2017) convention. Order matches _CSV_PROP_COLS.
_CONVERSIONS = np.array([
    1.0, 1.0, HAR2EV, HAR2EV, 1.0, HAR2EV,
    HAR2EV, HAR2EV, HAR2EV, HAR2EV, 1.0,
], dtype=np.float32)

_CSV_PROP_COLS = ["mu", "alpha", "homo", "lumo", "r2",
                  "zpve", "u0", "u298", "h298", "g298", "cv"]

# Atom types and hybridizations present in QM9
_ATOM_TYPES  = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}   # H, C, N, O, F → index 0-4
N_ATOM_FEAT  = 11


# ---------------------------------------------------------------------------
# Atom feature extraction
# ---------------------------------------------------------------------------

def _atom_features(atom) -> List[float]:
    """
    Extract 11-dim atom features matching the PyG QM9.x format used by
    Gilmer et al. (2017) and subsequent MTL benchmarks (Nash-MTL, FAMO, AIM).

    Requires the molecule to be sanitized (so hybridization / aromaticity
    fields are populated).  Falls back to atomic-number features only if
    sanitization was skipped.
    """
    from rdkit.Chem import rdchem

    # [0:5] Atom type one-hot: H, C, N, O, F
    type_oh = [0.0] * 5
    type_idx = _ATOM_TYPES.get(atom.GetAtomicNum(), 0)
    type_oh[type_idx] = 1.0

    # [5:8] Hybridization one-hot: SP, SP2, SP3
    _hyb = {
        rdchem.HybridizationType.SP:  0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
    }
    hyb_oh = [0.0, 0.0, 0.0]
    hyb_idx = _hyb.get(atom.GetHybridization(), -1)
    if hyb_idx >= 0:
        hyb_oh[hyb_idx] = 1.0

    # [8] aromatic   [9] formal charge   [10] total H count
    return type_oh + hyb_oh + [
        float(atom.GetIsAromatic()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
    ]


# ---------------------------------------------------------------------------
# Molecule data container
# ---------------------------------------------------------------------------

class _MolData:
    """Lightweight molecule container: atom features, 3-D positions, targets."""
    __slots__ = ("z", "x", "pos", "y")

    def __init__(self, z, x, pos, y):
        self.z   = z    # LongTensor   [n_atoms]        atomic numbers
        self.x   = x    # FloatTensor  [n_atoms, 11]    MPNN node features
        self.pos = pos  # FloatTensor  [n_atoms, 3]     3-D coordinates
        self.y   = y    # FloatTensor  [1, 11]          all 11 QM9 properties


# ---------------------------------------------------------------------------
# QM9 dataset
# ---------------------------------------------------------------------------

class QM9Lazy(Dataset):
    """
    QM9 dataset backed by raw SDF/CSV files, with per-atom MPNN node
    features (x). Each molecule is sanitized on first access so
    hybridization and aromaticity are correctly computed, then cached in
    memory (~20-30k molecules for the 10k-train + 10k-test splits used
    here — a few hundred MB at most) so repeated epochs over the same
    subset don't re-run RDKit parsing/sanitization every time. Without
    this, training for the paper's 300-400 epochs would re-sanitize the
    same molecules on every single epoch — the dominant (CPU-bound, not
    GPU-bound) cost of a full run.
    """

    def __init__(self, root: str):
        self.root  = Path(root)
        self._cache: Dict[int, "_MolData"] = {}
        raw_dir    = self.root / "raw"

        csv_path = raw_dir / "gdb9.sdf.csv"
        df = pd.read_csv(csv_path, usecols=["mol_id"] + _CSV_PROP_COLS)
        raw_props = df[_CSV_PROP_COLS].values.astype(np.float32)
        self._targets = torch.from_numpy(raw_props * _CONVERSIONS)

        excluded = set()
        unchar_path = raw_dir / "uncharacterized.txt"
        if unchar_path.exists():
            with open(unchar_path) as f:
                for line in f:
                    m = re.match(r"^\s*(\d+)\s", line)
                    if m:
                        excluded.add(int(m.group(1)) - 1)

        n_total = len(self._targets)
        self._valid_idx = [i for i in range(n_total) if i not in excluded]

        try:
            from rdkit import Chem
            sdf_path = str(raw_dir / "gdb9.sdf")
            # sanitize=False at load time; we sanitize per-molecule below
            self._supplier = Chem.SDMolSupplier(sdf_path, removeHs=False,
                                                 sanitize=False)
        except ImportError:
            raise ImportError("RDKit is required. Install with: "
                              "conda install -c conda-forge rdkit")

    def __len__(self) -> int:
        return len(self._valid_idx)

    def __getitem__(self, idx: int) -> _MolData:
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        from rdkit import Chem
        mol_idx = self._valid_idx[idx]
        mol     = self._supplier[mol_idx]

        if mol is None:
            raise ValueError(f"RDKit returned None for SDF index {mol_idx}.")

        # Sanitize so hybridization, aromaticity, numHs are correctly set
        Chem.SanitizeMol(mol, catchErrors=True)

        conf = mol.GetConformer()
        z    = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.long,
        )
        x    = torch.tensor(
            [_atom_features(atom) for atom in mol.GetAtoms()],
            dtype=torch.float32,
        )                                                         # [n, 11]
        pos  = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        y    = self._targets[mol_idx].unsqueeze(0)

        data = _MolData(z=z, x=x, pos=pos, y=y)
        self._cache[idx] = data
        return data


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_qm9(root: str = "../../data/qm9") -> QM9Lazy:
    root_path = Path(root)
    sdf_path  = root_path / "raw" / "gdb9.sdf"
    csv_path  = root_path / "raw" / "gdb9.sdf.csv"

    if not sdf_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"QM9 raw files not found in {root_path / 'raw'}.\n"
            "Run download_data.py first."
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
      guide_frac × n_train → guidance set (AIM policy loss only)
      val_frac   × n_train → validation set
      remainder            → primary training set
      up to 10 000         → held-out test set
    """
    rng     = np.random.default_rng(seed)
    N       = len(dataset)
    perm    = rng.permutation(N)
    n_guide = int(n_train * guide_frac)
    n_val   = int(n_train * val_frac)
    n_prim  = n_train - n_guide - n_val
    n_test  = min(10_000, N - n_train)

    assert n_prim > 0
    assert n_train + n_test <= N

    return {
        "primary":  Subset(dataset, perm[:n_prim].tolist()),
        "guidance": Subset(dataset, perm[n_prim: n_prim + n_guide].tolist()),
        "val":      Subset(dataset, perm[n_prim + n_guide: n_train].tolist()),
        "test":     Subset(dataset, perm[n_train: n_train + n_test].tolist()),
    }


def compute_normalization(
    subset: Subset,
    target_cols: List[int] = TARGET_COLS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_y = torch.cat([data.y[:, target_cols] for data in subset], dim=0)
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
    target_cols: List[int] = TARGET_COLS,
) -> dict:
    """
    Build DataLoaders for all splits using GNNCollator.

    Every batch has keys:
        x       : FloatTensor [B, T, 11]  node features (padded)
        dist    : FloatTensor [B, T, T]   pairwise distances
        mask    : BoolTensor  [B, T]        True = real atom
        targets : FloatTensor [B, N_TASKS]  (physical units, un-normalised)
        n_atoms : LongTensor  [B]
    """
    dataset = load_qm9(root)
    splits  = make_splits(dataset, n_train=n_train, seed=seed)
    collator = GNNCollator(target_cols=target_cols)

    loader_kw = dict(collate_fn=collator, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=False)

    return {
        "primary_loader": DataLoader(splits["primary"],  shuffle=True,  **loader_kw),
        "guide_loader":   DataLoader(splits["guidance"], shuffle=True,  **loader_kw),
        "val_loader":     DataLoader(splits["val"],      shuffle=False, **loader_kw),
        "test_loader":    DataLoader(splits["test"],     shuffle=False, **loader_kw),
        "means":          compute_normalization(splits["primary"], target_cols)[0],
        "stds":           compute_normalization(splits["primary"], target_cols)[1],
        "split_sizes":    {k: len(v) for k, v in splits.items()},
    }


# ---------------------------------------------------------------------------
# Quick inspection helper
# ---------------------------------------------------------------------------

def inspect_qm9(root: str = "../../data/qm9"):
    dataset = load_qm9(root)
    sample  = dataset[0]
    print(f"QM9 size      : {len(dataset):,} molecules")
    print(f"data.x shape  : {sample.x.shape}   (11-dim MPNN atom features)")
    print(f"data.pos shape: {sample.pos.shape}  (3-D coordinates)")
    print(f"Tasks         : {TASK_NAMES}  (cols {TARGET_COLS})")
    return dataset


if __name__ == "__main__":
    inspect_qm9()
    loaders = get_loaders(n_train=10_000)
    print("\nSplit sizes:", loaders["split_sizes"])
    batch = next(iter(loaders["primary_loader"]))
    print("Batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s}: {v.shape}")
