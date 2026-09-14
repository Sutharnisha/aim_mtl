"""
UniMolCollator — converts a batch of QM9 PyG Data objects into the
padded tensor format expected by the Uni-Mol SE(3)-Transformer encoder.

Uni-Mol input tensors (all batch-first):
  src_tokens   : LongTensor  [B, T]        atom-type token ids (padded with the dictionary pad id)
  src_coord    : FloatTensor [B, T, 3]     3-D coordinates     (padded with 0)
  src_distance : FloatTensor [B, T, T]     pairwise distances  (padded with 0)
  src_edge_type: LongTensor  [B, T, T]     atom-pair type ids  (padded with 0)

where T = max atoms in the batch + 2  ([CLS] prepended, [SEP] appended).

Input construction mirrors unimol_tools' own `coords2unimol` preprocessing
(the code the pretrained weights were trained with):
  tokens   = [CLS] + atoms + [SEP]
  coords   = centred (minus the molecule's mean), zeros for [CLS]/[SEP]
  distance = pairwise distances over the full token sequence
  edge     = token_i * len(dictionary) + token_j  (full dictionary length,
             NOT just the elements present in QM9)

Atom type vocabulary:
  Loaded from the pretrained model's own mol.dict.txt via unimol_tools
  (see `get_vocab`); a byte-identical fallback is used if the package is
  missing. Special ids: [CLS]=0 [UNK]=1 [PAD]=2 [SEP]=3, elements from 4.

Edge type encoding:
  For each atom pair (i, j): edge_type = token_i * len(dictionary) + token_j
  This is the standard Uni-Mol edge-type encoding used in the original paper.
  len(dictionary) is the FULL pretrained vocabulary size (special tokens +
  every element in Uni-Mol's mol.dict.txt), so the edge-type ids match the
  pretrained edge-type embedding table.
"""

import torch
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# Atom vocabulary — MUST be the pretrained model's own dictionary
# ---------------------------------------------------------------------------
#
# UniMolModel builds its token / edge-type embedding tables from
# unimol_tools' `mol.dict.txt` loaded through unimol_tools' Dictionary class,
# which assigns the special tokens in the constructor order
#     [CLS]=0  [UNK]=1  [PAD]=2  [SEP]=3
# then the file's element symbols in file order (C=4, N=5, O=6, S=7, H=8,
# Cl=9, F=10, ...), then appends [MASK] (index 30) -> len(dictionary) = 31
# and edge_types = 31*31 = 961. The collator must reproduce EXACTLY these
# ids or the pretrained embeddings are looked up with the wrong rows
# (and the padding mask uses the wrong pad index).
#
# We therefore load the real dictionary through unimol_tools whenever it is
# importable (downloading mol.dict.txt on first use, same as the model
# does) and fall back to a byte-for-byte replica of that file otherwise.

# Exact contents of unimol_tools/weights/mol.dict.txt (unimol-tools 0.1.6)
_MOL_DICT_TXT = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]",
    "C", "N", "O", "S", "H", "Cl", "F", "Br", "I", "Si", "P", "B",
    "Na", "K", "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr", "Se",
    "Gd", "Au", "Li",
]

_QM9_ATOMIC_NUM_TO_SYMBOL = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 33: "As", 34: "Se", 35: "Br", 53: "I",
}


class _Vocab:
    """Token ids of the pretrained Uni-Mol dictionary (incl. [MASK])."""
    __slots__ = ("sym2idx", "bos", "pad", "eos", "unk", "size", "source")

    def __init__(self, sym2idx, bos, pad, eos, unk, size, source):
        self.sym2idx, self.bos, self.pad, self.eos, self.unk = sym2idx, bos, pad, eos, unk
        self.size, self.source = size, source

    def index(self, symbol: str) -> int:
        return self.sym2idx.get(symbol, self.unk)


def _vocab_from_unimol_tools() -> Optional[_Vocab]:
    """Load the dictionary exactly the way UniMolModel does."""
    try:
        import os
        from unimol_tools.data.dictionary import Dictionary
        from unimol_tools.weights.weighthub import get_weight_dir, weight_download
        weight_dir = get_weight_dir()
        path = os.path.join(weight_dir, "mol.dict.txt")
        if not os.path.exists(path):
            weight_download("mol.dict.txt", weight_dir)
        d = Dictionary.load(path)
        d.add_symbol("[MASK]", is_special=True)          # as UniMolModel.__init__
        return _Vocab({sym: d.index(sym) for sym in d.symbols},
                      d.bos(), d.pad(), d.eos(), d.unk(), len(d), "unimol_tools")
    except Exception:
        return None


def _vocab_fallback() -> _Vocab:
    """Replica of unimol_tools.Dictionary(mol.dict.txt) + [MASK] without the package."""
    symbols: List[str] = []
    for tok in ["[CLS]", "[UNK]", "[PAD]", "[SEP]"] + _MOL_DICT_TXT + ["[MASK]"]:
        if tok not in symbols:
            symbols.append(tok)
    sym2idx = {sym: i for i, sym in enumerate(symbols)}
    return _Vocab(sym2idx, sym2idx["[CLS]"], sym2idx["[PAD]"], sym2idx["[SEP]"],
                  sym2idx["[UNK]"], len(symbols), "fallback")


_VOCAB: Optional[_Vocab] = None


def get_vocab() -> _Vocab:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _vocab_from_unimol_tools() or _vocab_fallback()
    return _VOCAB


def get_symbol_to_idx() -> dict:
    return get_vocab().sym2idx


def atomic_num_to_token(z: int) -> int:
    """Map an atomic number to the pretrained Uni-Mol vocabulary index."""
    v = get_vocab()
    return v.index(_QM9_ATOMIC_NUM_TO_SYMBOL.get(z, "[UNK]"))


def vocab_size() -> int:
    """len(dictionary) incl. [MASK] — the edge-type multiplier (31 for mol.dict.txt)."""
    return get_vocab().size


# ---------------------------------------------------------------------------
# Core collation logic
# ---------------------------------------------------------------------------

def _pad_1d(tensors: List[torch.Tensor], pad_val: int = 0) -> torch.Tensor:
    """Pad a list of 1-D tensors to the same length."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


def _pad_2d(tensors: List[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
    """Pad a list of 2-D tensors [L, D] to [B, max_L, D]."""
    max_len = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    out = torch.full((len(tensors), max_len, D), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0], :] = t
    return out


def _pad_3d(tensors: List[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
    """Pad a list of 2-D square tensors [L, L] to [B, max_L, max_L]."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len, max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        out[i, :L, :L] = t
    return out


class UniMolCollator:
    """
    Callable collator for torch.utils.data.DataLoader.

    Converts a list of _MolData objects (from data.py's QM9Lazy dataset,
    each exposing .z, .pos, .y) into a dict of padded tensors ready for
    the Uni-Mol encoder.

    Usage:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32,
                            collate_fn=UniMolCollator(target_cols=list(range(11))))
    """

    def __init__(self, target_cols: List[int] = list(range(11))):
        self.target_cols = target_cols

    def __call__(self, data_list) -> dict:
        """
        Args:
            data_list: list of _MolData objects

        Returns:
            dict with keys:
              src_tokens, src_coord, src_distance, src_edge_type  — encoder inputs
              targets    : FloatTensor [B, n_tasks]
              n_atoms    : LongTensor  [B]           number of real atoms (excl. CLS/SEP)
        """
        v        = get_vocab()
        voc_size = v.size                # len(dictionary) incl. [MASK], as in unimol_tools

        tokens_list    = []
        coord_list     = []
        distance_list  = []
        edge_type_list = []
        targets_list   = []
        n_atoms_list   = []

        for data in data_list:
            z   = data.z.numpy()            # [n_atoms]  atomic numbers
            pos = data.pos.numpy()          # [n_atoms, 3]
            n   = len(z)

            # ── Token ids ────────────────────────────────────────────────
            atom_tokens = np.array([atomic_num_to_token(int(zi)) for zi in z], dtype=np.int64)
            # [CLS] + atoms + [SEP]  (bos ... eos, exactly as unimol_tools coords2unimol)
            tokens = np.concatenate([[v.bos], atom_tokens, [v.eos]])       # [n+2]

            # ── Coordinates: centre on the molecule, zeros for CLS/SEP ────
            pos    = (pos - pos.mean(axis=0, keepdims=True)).astype(np.float32)
            zero_c = np.zeros((1, 3), dtype=np.float32)
            coords = np.concatenate([zero_c, pos, zero_c], axis=0)         # [n+2, 3]

            # ── Pairwise distances over the full token sequence ───────────
            diff = coords[:, None, :] - coords[None, :, :]                 # [n+2, n+2, 3]
            dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)    # [n+2, n+2]

            # ── Edge type = token_i * vocab_size + token_j ────────────────
            edge_type = (
                tokens[:, None] * voc_size + tokens[None, :]
            ).astype(np.int64)                                    # [n+2, n+2]

            # ── Targets ───────────────────────────────────────────────────
            y = data.y[0, self.target_cols].numpy().astype(np.float32)  # [n_tasks]

            tokens_list.append(torch.tensor(tokens,    dtype=torch.long))
            coord_list.append(torch.tensor(coords,     dtype=torch.float32))
            distance_list.append(torch.tensor(dist,    dtype=torch.float32))
            edge_type_list.append(torch.tensor(edge_type, dtype=torch.long))
            targets_list.append(torch.tensor(y,        dtype=torch.float32))
            n_atoms_list.append(n)

        # ── Pad all sequences to max length in batch ──────────────────────
        src_tokens    = _pad_1d(tokens_list,   pad_val=v.pad)       # [B, T]
        src_coord     = _pad_2d(coord_list,    pad_val=0.0)         # [B, T, 3]
        src_distance  = _pad_3d(distance_list, pad_val=0.0)         # [B, T, T]
        src_edge_type = _pad_3d(edge_type_list,pad_val=0)           # [B, T, T]
        targets       = torch.stack(targets_list, dim=0)            # [B, n_tasks]
        n_atoms       = torch.tensor(n_atoms_list, dtype=torch.long)# [B]

        return {
            "src_tokens":    src_tokens,
            "src_coord":     src_coord,
            "src_distance":  src_distance,
            "src_edge_type": src_edge_type,
            "targets":       targets,
            "n_atoms":       n_atoms,
        }


# ---------------------------------------------------------------------------
# Helper to move a collated batch to a device
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in the collated batch dict to the given device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
