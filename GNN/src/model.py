"""
MPNN multi-task molecular model — Gilmer et al. (2017) backbone + 3 MLP heads.

Architecture: Neural Message Passing Network (MPNN)
  Gilmer et al. "Neural Message Passing for Quantum Chemistry", ICML 2017.
  This is the backbone used by Nash-MTL → FAMO → AIM paper chain.
──────────────────────────────────────────────────────────────────────────────
  Node projection : Linear(11, hidden_dim=64)
       ↓  [B, N, 64]
  T=3 × NNConv + GRU interaction steps:
       ─ RBF expansion of d_ij  →  [B, N, N, n_rbf=16]
       ─ Edge network (MLP):  n_rbf → 128 → hidden_dim²
         produces weight matrix W_ij ∈ R^{d×d} for each atom pair
       ─ Message:   m_{i←j}  = W_ij  @  h_j^t          (matrix-vector)
       ─ Aggregate: m_i       = mean_{j ∈ N(i)}  m_{i←j}
       ─ Update:    h_i^{t+1} = GRUCell( m_i ,  h_i^t )
       ↓  [B, N, 64]
  Set2Set readout (processing_steps=3)
       ↓  [B, 128]   (= [B, 2 × hidden_dim])
  N × task-specific MLP head (N=11, full QM9 property set):
       128 → head_hidden → head_hidden//2 → 1
──────────────────────────────────────────────────────────────────────────────

NNConv uses a weight matrix W_ij ∈ R^{d×d} per pair so the message is a
full linear transform of the source atom's features (more expressive than
element-wise gating).

Set2Set (Vinyals et al. 2015) is a learned attention-based graph-level
readout that is permutation invariant and more powerful than mean-pooling.

hidden_dim = 64  (as in Nash-MTL / FAMO / AIM benchmark setup)
Task head input = 2 × hidden_dim = 128  (Set2Set concatenation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# ---------------------------------------------------------------------------
# Global hyper-parameters  (matching Nash-MTL / FAMO benchmark setup)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64     # node hidden dimension
N_RBF      = 16     # Gaussian RBF centres for edge-distance encoding
N_STEPS    = 3      # number of message-passing iterations (T)
CUTOFF     = 10.0   # Angstrom — pairs beyond this are masked


# ---------------------------------------------------------------------------
# Gaussian RBF expansion of distances
# ---------------------------------------------------------------------------

class RBFExpansion(nn.Module):
    """
    Gaussian radial basis functions for distance encoding.

      e_k(d) = exp( -γ · (d − μ_k)² )

    μ_k : N_RBF centres evenly spaced in [0, cutoff]
    γ   : 0.5 / spacing²
    """

    def __init__(self, n_rbf: int = N_RBF, cutoff: float = CUTOFF):
        super().__init__()
        centres = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centres", centres)
        spacing    = cutoff / max(n_rbf - 1, 1)
        self.gamma = 0.5 / (spacing ** 2)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist [...] → [..., n_rbf]"""
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centres) ** 2)


# ---------------------------------------------------------------------------
# NNConv + GRU interaction step
# ---------------------------------------------------------------------------

class NNConvGRUBlock(nn.Module):
    """
    One MPNN interaction step (Gilmer et al. Eq. 1):

        M_t(h_v, h_w, e_{vw}) = A(e_{vw}) · h_w
        m_v^{t+1} = mean_{w ∈ N(v)} M_t(...)
        h_v^{t+1} = GRU( m_v^{t+1}, h_v^t )

    A(e_{vw}) is the edge network that maps the n_rbf-dim RBF edge feature
    to a d×d weight matrix (Gilmer et al. "edge network" variant).

    Aggregation: mean over neighbors (edges with pair_mask=True).
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_rbf: int = N_RBF):
        super().__init__()
        # Edge network: n_rbf → 128 → hidden_dim²
        self.edge_net = nn.Sequential(
            nn.Linear(n_rbf, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim * hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        h:         torch.Tensor,   # [B, N, d]
        rbf:       torch.Tensor,   # [B, N, N, n_rbf]
        pair_mask: torch.Tensor,   # [B, N, N] bool — True = valid pair
    ) -> torch.Tensor:
        """Returns updated node features [B, N, d]."""
        B, N, d = h.shape

        # ── Edge weight matrices ──────────────────────────────────────────
        # W_flat: [B, N, N, d*d]  →  W: [B*N*N, d, d]
        W = self.edge_net(rbf).view(B * N * N, d, d)   # [B*N*N, d, d]

        # ── Messages  m_{i←j} = W_{ij} @ h_j ────────────────────────────
        # h_j expanded over target-atom (i) dimension: [B, N, N, d]
        # → contiguous reshape needed before view
        h_j = h.unsqueeze(1).expand(B, N, N, d).reshape(B * N * N, d, 1)
        #                              ^broadcast over i

        # Batched matmul: [B*N*N, d, d] × [B*N*N, d, 1] → [B*N*N, d]
        msg = torch.bmm(W, h_j).squeeze(-1).view(B, N, N, d)  # [B, N, N, d]

        # ── Mask invalid pairs ────────────────────────────────────────────
        msg = msg * pair_mask.unsqueeze(-1).float()            # [B, N, N, d]

        # ── Mean aggregation over source atoms j ─────────────────────────
        n_nbr = pair_mask.float().sum(dim=2, keepdim=True).clamp(min=1.0)
        agg   = msg.sum(dim=2) / n_nbr                         # [B, N, d]

        # ── GRU update (per-atom, flattened over B and N) ─────────────────
        h_new = self.gru(
            agg.view(B * N, d),      # input:  aggregated message
            h.view(B * N, d),        # hidden: current features
        ).view(B, N, d)              # [B, N, d]

        return h_new


# ---------------------------------------------------------------------------
# Set2Set readout  (Vinyals et al. 2015)
# ---------------------------------------------------------------------------

class Set2Set(nn.Module):
    """
    Permutation-invariant graph readout via learned attention over T steps.

    Algorithm:
      q^0 = zeros [B, 2d]
      for t in 1..T:
          q_t, lstm_h = LSTM(q^{t-1})         →  q_t [B, d]
          e_i = q_t · h_i  / sqrt(d)          →  [B, N]  attention energy
          a_i = softmax(e_i, masked)           →  [B, N]  attention weights
          r_t = Σ_i a_i h_i                   →  [B, d]  readout
          q^t = cat(q_t, r_t)                 →  [B, 2d]
      return q^T                              →  [B, 2d]
    """

    def __init__(self, in_channels: int, processing_steps: int = 3):
        super().__init__()
        self.in_channels      = in_channels
        self.out_channels     = 2 * in_channels
        self.processing_steps = processing_steps
        self.scale            = in_channels ** -0.5
        # LSTM input = current query (2d), output = next query (d)
        self.lstm = nn.LSTMCell(self.out_channels, in_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : [B, N, in_channels]   node features
            mask : [B, N]                True = real atom
        Returns:
            [B, 2 * in_channels]         molecular embedding
        """
        B, N, d = x.shape

        # Initialise LSTM hidden/cell states and query
        lstm_h = x.new_zeros(B, d)
        lstm_c = x.new_zeros(B, d)
        q_star = x.new_zeros(B, self.out_channels)

        for _ in range(self.processing_steps):
            # LSTM step: takes previous query, outputs new hidden state
            lstm_h, lstm_c = self.lstm(q_star, (lstm_h, lstm_c))  # [B, d]

            # Attention energy: dot product of query with each node
            e = (x * lstm_h.unsqueeze(1)).sum(dim=-1) * self.scale  # [B, N]

            # Mask padding atoms before softmax
            e = e.masked_fill(~mask, float('-inf'))
            a = torch.softmax(e, dim=-1)                            # [B, N]
            # Fix NaN when all atoms are padding (shouldn't happen in practice)
            a = torch.nan_to_num(a, nan=0.0)

            # Attended readout
            r = (a.unsqueeze(-1) * x).sum(dim=1)                    # [B, d]

            # New query = cat(lstm output, readout)
            q_star = torch.cat([lstm_h, r], dim=-1)                 # [B, 2d]

        return q_star   # [B, 2d]


# ---------------------------------------------------------------------------
# MPNN Encoder
# ---------------------------------------------------------------------------

class MPNNEncoder(nn.Module):
    """
    Full MPNN encoder:
      11-dim atom features → node projection → T NNConv+GRU steps
      → Set2Set readout → [B, 2*hidden_dim]
    """

    HIDDEN_DIM = HIDDEN_DIM

    def __init__(
        self,
        hidden_dim:       int   = HIDDEN_DIM,
        n_steps:          int   = N_STEPS,
        n_rbf:            int   = N_RBF,
        cutoff:           float = CUTOFF,
        n_atom_feat:      int   = 11,
        set2set_steps:    int   = 3,
        trainable_layers: int   = -1,
    ):
        """
        trainable_layers:
            -1 : all layers trainable (default for from-scratch training)
             k : freeze first (n_steps - k) NNConv blocks; keep last k.
        """
        super().__init__()
        self.cutoff  = cutoff
        self.n_steps = n_steps

        self.node_proj = nn.Linear(n_atom_feat, hidden_dim)
        self.rbf       = RBFExpansion(n_rbf=n_rbf, cutoff=cutoff)
        self.blocks    = nn.ModuleList([
            NNConvGRUBlock(hidden_dim=hidden_dim, n_rbf=n_rbf)
            for _ in range(n_steps)
        ])
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)

        # Selective freezing: keep only the last `trainable_layers` NNConv blocks
        if trainable_layers >= 0:
            for p in self.node_proj.parameters():
                p.requires_grad_(False)
            n_freeze = n_steps - trainable_layers
            for block in self.blocks[:n_freeze]:
                for p in block.parameters():
                    p.requires_grad_(False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch : dict with keys
                x    [B, N, 11]    atom features
                dist [B, N, N]     pairwise distances (Å)
                mask [B, N]        True = real atom

        Returns:
            FloatTensor [B, 2*hidden_dim]   molecular embedding
        """
        x    = batch["x"]      # [B, N, 11]
        dist = batch["dist"]   # [B, N, N]
        mask = batch["mask"]   # [B, N]

        # Initial node embedding
        h = self.node_proj(x)  # [B, N, hidden_dim]

        # Gaussian RBF expansion of distances
        rbf = self.rbf(dist)   # [B, N, N, n_rbf]

        # Pair validity: both atoms real and within cutoff
        pair_mask = (
            mask.unsqueeze(2) &        # target atom i real
            mask.unsqueeze(1) &        # source atom j real
            (dist < self.cutoff)       # within interaction range
        )   # [B, N, N]

        # T message-passing steps
        for block in self.blocks:
            h = block(h, rbf, pair_mask)   # [B, N, hidden_dim]

        # Set2Set readout → [B, 2*hidden_dim]
        return self.set2set(h, mask)


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------

class MPNNMultiTask(nn.Module):
    """
    Full multi-task model:
        MPNNEncoder  →  [B, 2*hidden_dim = 128]
        N × MLP head →  [B, 1]

    Interface identical to UniMolMultiTask so train.py is unchanged:
      shared_forward(batch) → [B, 2*hidden_dim]
      task_forward(h, i)    → [B]
      shared_params()       → trainable encoder params (AIM gradient hooks)
    """

    def __init__(
        self,
        n_tasks:          int   = 11,
        head_hidden:      int   = 64,
        hidden_dim:       int   = HIDDEN_DIM,
        n_steps:          int   = N_STEPS,
        n_rbf:            int   = N_RBF,
        cutoff:           float = CUTOFF,
        trainable_layers: int   = -1,
    ):
        super().__init__()
        self.n_tasks = n_tasks

        self.encoder = MPNNEncoder(
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            n_rbf=n_rbf,
            cutoff=cutoff,
            trainable_layers=trainable_layers,
        )
        enc_out = 2 * hidden_dim   # Set2Set output dimension

        # Task heads: 128 → head_hidden → head_hidden//2 → 1
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_out, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.ReLU(),
                nn.Linear(head_hidden // 2, 1),
            )
            for _ in range(n_tasks)
        ])

    def shared_forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(batch)   # [B, 2*hidden_dim]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.shared_forward(batch)
        return torch.cat([head(h) for head in self.heads], dim=1)

    def task_forward(self, h: torch.Tensor, task_idx: int) -> torch.Tensor:
        return self.heads[task_idx](h).squeeze(-1)

    def shared_params(self) -> List[nn.Parameter]:
        """Trainable encoder parameters — AIM/PCGrad intervene on their gradients."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_params(self, task_idx: int) -> List[nn.Parameter]:
        return list(self.heads[task_idx].parameters())

    def param_count(self) -> dict:
        enc_total     = sum(p.numel() for p in self.encoder.parameters())
        enc_trainable = sum(p.numel() for p in self.encoder.parameters()
                           if p.requires_grad)
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
    hidden_dim:       int   = HIDDEN_DIM,
    n_steps:          int   = N_STEPS,
    trainable_layers: int   = -1,
) -> MPNNMultiTask:

    model = MPNNMultiTask(
        n_tasks=n_tasks,
        head_hidden=head_hidden,
        hidden_dim=hidden_dim,
        n_steps=n_steps,
        trainable_layers=trainable_layers,
    ).to(device)

    counts = model.param_count()
    tl_str = "all" if trainable_layers == -1 else f"last {trainable_layers}"
    print(
        f"Model : MPNN (Gilmer 2017)  {n_tasks} heads  "
        f"hidden_dim={hidden_dim}  n_steps={n_steps}  "
        f"Set2Set readout\n"
        f"  Encoder total      : {counts['encoder_total']:,}\n"
        f"  Encoder trainable  : {counts['encoder_trainable']:,}"
        f"  ({tl_str} blocks)\n"
        f"  Head params        : {counts['heads']:,}  ({n_tasks} task heads)\n"
        f"  Total trainable    : {counts['total_trainable']:,}"
    )
    return model
