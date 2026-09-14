"""
GNNCollator — packs a batch of QM9 _MolData objects into padded tensors
for the MPNN (Gilmer et al. 2017) encoder.

Batch tensors:
  x       : FloatTensor [B, T, 11]    atom features (padded with 0)
  dist    : FloatTensor [B, T, T]     pairwise Euclidean distances (Å)
  mask    : BoolTensor  [B, T]         True = real atom, False = padding
  targets : FloatTensor [B, n_tasks]  property targets (physical units)
  n_atoms : LongTensor  [B]            real atom count per molecule

T = max atoms in the current batch (dynamic padding).
No [CLS] token — the MPNN uses Set2Set as its readout.
"""

import torch
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _pad_x(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad [n, 11] atom-feature tensors to [B, max_n, 11]."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.zeros(len(tensors), max_len, tensors[0].shape[1],
                      dtype=torch.float32)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _pad_dist(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad [n, n] distance matrices to [B, max_n, max_n]."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.zeros(len(tensors), max_len, max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L, :L] = t
    return out


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class GNNCollator:
    """
    Callable collator for PyTorch DataLoader.

    Converts _MolData objects (from QM9Lazy) into padded dense tensors
    ready for the MPNN encoder.

    The dense all-pair distance matrix avoids any graph-library dependency
    and is compatible with the all-pair interaction pattern used by the MPNN
    encoder (pair_mask applied inside NNConvGRUBlock).
    """

    def __init__(self, target_cols: List[int] = list(range(11))):
        self.target_cols = target_cols

    def __call__(self, data_list) -> dict:
        """
        Args:
            data_list : list of _MolData  (z, x, pos, y attributes)

        Returns dict with keys:
            x       : FloatTensor [B, T, 11]   atom features
            dist    : FloatTensor [B, T, T]    pairwise distances (Å)
            mask    : BoolTensor  [B, T]        True = real atom
            targets : FloatTensor [B, n_tasks]
            n_atoms : LongTensor  [B]
        """
        x_list      = []
        dist_list   = []
        target_list = []
        n_atoms_list = []

        for data in data_list:
            x   = data.x        # FloatTensor [n, 11]
            pos = data.pos      # FloatTensor [n, 3]
            n   = len(x)

            # Pairwise Euclidean distances
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # [n, n, 3]
            dist = diff.norm(dim=-1)                      # [n, n]

            y = data.y[0, self.target_cols]               # [n_tasks]

            x_list.append(x)
            dist_list.append(dist)
            target_list.append(y)
            n_atoms_list.append(n)

        src_x    = _pad_x(x_list)          # [B, T, 11]
        src_dist = _pad_dist(dist_list)    # [B, T, T]

        T    = src_x.shape[1]
        mask = torch.zeros(len(data_list), T, dtype=torch.bool)
        for i, n in enumerate(n_atoms_list):
            mask[i, :n] = True

        targets = torch.stack(target_list, dim=0)
        n_atoms = torch.tensor(n_atoms_list, dtype=torch.long)

        return {
            "x":       src_x,
            "dist":    src_dist,
            "mask":    mask,
            "targets": targets,
            "n_atoms": n_atoms,
        }


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in the collated batch to the given device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
