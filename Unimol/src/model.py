"""
Multi-task molecular model — Uni-Mol backbone + 3 MLP prediction heads (mu, U0, U).

Architecture (Uni-Mol SE(3)-Transformer):
  Uni-Mol SE(3)-Transformer encoder (pretrained, ~47M params)
    15 transformer blocks
    hidden_dim  = 512
    num_heads   = 8
    head_dim    = 64   (512 / 8 = 64 per head)
      ↓  [CLS] token embedding  [B, 512]
  3 independent task-specific MLP heads
      ↓  per-task scalar prediction

Uni-Mol pretrained weights are downloaded automatically by unimol_tools
on first run (~700 MB, stored in ~/.unimol/).

By default the full backbone is fine-tuned (trainable_layers=-1); pass
--trainable_layers N to freeze all but the last N transformer blocks, or
--freeze_backbone to train only the task heads.

Input format (produced by UniMolCollator in unimol_collate.py):
  src_tokens   : LongTensor  [B, T]        atom-type token ids (padded)
  src_coord    : FloatTensor [B, T, 3]     3-D coordinates (padded)
  src_distance : FloatTensor [B, T, T]     pairwise distances (padded)
  src_edge_type: LongTensor  [B, T, T]     atom-pair type ids (padded)

Training protocol (learning-rate schedulers, patience, warm restarts) is
handled in train.py so the architecture remains unchanged.
"""

import torch
import torch.nn as nn
from typing import Dict, List


# ---------------------------------------------------------------------------
# Uni-Mol encoder wrapper
# ---------------------------------------------------------------------------

class UniMolEncoder(nn.Module):
    """
    Loads pretrained Uni-Mol weights via unimol_tools and exposes the
    transformer encoder as a standard nn.Module for gradient hook access.

    The backbone is accessed as self.backbone — AIM collects gradients
    w.r.t. list(self.backbone.parameters()).

    Returns [CLS] token embedding: FloatTensor [B, hidden_dim].
    """

    HIDDEN_DIM = 512   # Uni-Mol standard hidden dimension

    def __init__(
        self,
        remove_hs:        bool = False,
        freeze_backbone:  bool = False,
        trainable_layers: int  = -1,
    ):
        """
        trainable_layers: number of last transformer blocks to keep trainable
          for AIM/PCGrad gradient computation.
          - 0  : freeze all backbone (only task heads train)
          - N  : freeze all but last N blocks
          - -1 : keep all backbone trainable (default)
          freeze_backbone=True overrides trainable_layers to 0.
        """
        super().__init__()

        from unimol_tools import UniMolRepr
        _repr = UniMolRepr(data_type="molecule", remove_hs=remove_hs)

        if hasattr(_repr, "model"):
            self.backbone = _repr.model
        elif hasattr(_repr, "encoder"):
            self.backbone = _repr.encoder
        else:
            raise AttributeError(
                "Cannot find nn.Module inside UniMolRepr. "
                "Check your unimol_tools version."
            )

        # Freeze all backbone params first
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if not freeze_backbone and trainable_layers != 0:
            if trainable_layers == -1:
                # Unfreeze everything
                for p in self.backbone.parameters():
                    p.requires_grad_(True)
            else:
                # Unfreeze only the last N transformer blocks
                # backbone.encoder.layers is a ModuleList of 15 blocks
                blocks = self.backbone.encoder.layers
                for block in blocks[-trainable_layers:]:
                    for p in block.parameters():
                        p.requires_grad_(True)

        # unimol_tools attaches a placeholder `classification_head` to the
        # backbone. It is NOT on the path to the [CLS] representation we use,
        # so it must not be trainable: AIM / PCGrad differentiate every task
        # loss w.r.t. shared_params() and a parameter outside the graph makes
        # torch.autograd.grad raise. (513 params, irrelevant to the encoder.)
        if hasattr(self.backbone, "classification_head"):
            for p in self.backbone.classification_head.parameters():
                p.requires_grad_(False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict produced by UniMolCollator with keys
                   src_tokens, src_coord, src_distance, src_edge_type

        Returns:
            FloatTensor [B, 512]  — [CLS] token representation
        """
        # Argument order matches UniMolModel.forward signature exactly:
        #   (src_tokens, src_distance, src_coord, src_edge_type)
        # return_repr=True  →  returns cls_repr [B, hidden_dim] directly
        cls_repr = self.backbone(
            batch["src_tokens"],
            batch["src_distance"],   # distance BEFORE coord
            batch["src_coord"],
            batch["src_edge_type"],
            return_repr=True,
        )
        return cls_repr   # [B, 512]


# ---------------------------------------------------------------------------
# Multi-task model: Uni-Mol encoder + 11 MLP heads
# ---------------------------------------------------------------------------

class UniMolMultiTask(nn.Module):
    """
    Full multi-task model:
        UniMolEncoder  →  [B, 512]
        n_tasks × MLP head  →  [B, 1] each → cat → [B, n_tasks]

    Used by the AIM training loop:
      • shared_forward(batch) → [B, 512]   (one call per step)
      • task_forward(h, i)    → [B]        (one call per task for AIM grads)
      • shared_params()       → list of backbone params for gradient hooks
    """

    def __init__(
        self,
        n_tasks:          int   = 11,
        head_hidden:      int   = 64,
        remove_hs:        bool  = False,
        freeze_backbone:  bool  = False,
        trainable_layers: int   = -1,
    ):
        super().__init__()
        self.n_tasks = n_tasks

        self.encoder = UniMolEncoder(
            remove_hs=remove_hs,
            freeze_backbone=freeze_backbone,
            trainable_layers=trainable_layers,
        )
        enc_dim = UniMolEncoder.HIDDEN_DIM   # 512

        # n_tasks task-specific MLP heads: 512 → head_hidden → head_hidden//2 → 1
        # (same head architecture / width / activation as GNN/src/model.py)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.ReLU(),
                nn.Linear(head_hidden // 2, 1),
            )
            for _ in range(n_tasks)
        ])

    # -----------------------------------------------------------------------

    def shared_forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Uni-Mol [CLS] embedding. Returns [B, 512]."""
        return self.encoder(batch)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Full forward pass. Returns [B, n_tasks] predictions."""
        h = self.shared_forward(batch)
        return torch.cat([head(h) for head in self.heads], dim=1)

    def task_forward(self, h: torch.Tensor, task_idx: int) -> torch.Tensor:
        """Apply head[task_idx] to pre-computed representation. Returns [B]."""
        return self.heads[task_idx](h).squeeze(-1)

    # -----------------------------------------------------------------------
    # Parameter group helpers — used by AIM gradient collection
    # -----------------------------------------------------------------------

    def shared_params(self) -> List[nn.Parameter]:
        """Trainable backbone parameters — AIM/PCGrad intervene on their gradients."""
        return [p for p in self.encoder.backbone.parameters() if p.requires_grad]

    def head_params(self, task_idx: int) -> List[nn.Parameter]:
        return list(self.heads[task_idx].parameters())

    def all_head_params(self) -> List[nn.Parameter]:
        return list(self.heads.parameters())

    def param_count(self) -> dict:
        enc_total     = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_n        = sum(p.numel() for p in self.heads.parameters())
        return {
            "encoder_total":     enc_total,
            "encoder_trainable": enc_trainable,
            "heads":             head_n,
            "total_trainable":   enc_trainable + head_n,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    device:           torch.device,
    n_tasks:          int   = 11,
    head_hidden:      int   = 64,
    freeze_backbone:  bool  = False,
    trainable_layers: int   = -1,
    remove_hs:        bool  = False,
) -> UniMolMultiTask:

    print("Loading pretrained Uni-Mol weights...")
    model = UniMolMultiTask(
        n_tasks=n_tasks,
        head_hidden=head_hidden,
        freeze_backbone=freeze_backbone,
        trainable_layers=trainable_layers,
        remove_hs=remove_hs,
    ).to(device)

    counts = model.param_count()
    tl_str = "all" if trainable_layers == -1 else f"last {trainable_layers}"
    print(
        f"Model : Uni-Mol SE(3)-Transformer  {n_tasks} heads  head_hidden={head_hidden}\n"
        f"  Backbone total      : {counts['encoder_total']:,}\n"
        f"  Backbone trainable  : {counts['encoder_trainable']:,}  "
        f"({tl_str} transformer blocks)\n"
        f"  Head params         : {counts['heads']:,}  ({n_tasks} task heads)\n"
        f"  Total trainable     : {counts['total_trainable']:,}"
    )
    return model
