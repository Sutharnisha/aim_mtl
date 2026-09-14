"""
AIM: Adaptive Intervention for Deep Multi-task Learning of Molecular Properties
Minot & Schneider, NeurIPS AI4Science 2025  |  arXiv:2509.25955

Implements from paper equations:
  Eq 1 — Projection weight:
      w_proj(i,j) = sigma( k * (tau_ij - cos(g_i, g_j)) )

  Eq 2 — Modified gradient per task:
      g_i' = g_i - sum_{j != i} w_proj(i,j) * proj_{g_j}(g_i)
      where proj_{g_j}(g_i) = (g_i · g_j / ||g_j||^2) * g_j

  Eq 3 — Intervened update:
      g_intervened = sum_i g_i'
      theta_{t+1} <- Adam(theta_t, g_intervened)

  Eq 4 — Policy loss:
      L_policy = lambda_g * L_guide + lambda_m * L_magnitude + lambda_p * L_progress

  Eq 5 — Magnitude preservation:
      L_magnitude = ( ||g_intervened|| - sum_i ||g_i|| )^2

  Eq 6 — Progress on hard tasks:
      L_progress = -sum_i alpha_i * (g_intervened · g_i)
      where alpha_i = L_i / sum_j L_j

Policy variants:
  AIMScalarPolicy  — single global threshold tau (shared by all task pairs)
  AIMMatrixPolicy  — unique tau_ij per pair (N x N learnable matrix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core intervention function (shared by both policy variants)
# ---------------------------------------------------------------------------

def aim_intervention(
    gradients: List[torch.Tensor],
    tau: torch.Tensor,
    k: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply AIM gradient intervention (Equations 1-3).

    Args:
        gradients : List[Tensor[d]]  per-task flattened gradients (detached ok)
        tau       : Tensor[N, N]     learnable threshold matrix (differentiable)
        k         : float            temperature — sharpens the sigmoid gate

    Returns:
        g_intervened : Tensor[d]     sum of modified gradients (Eq 3)
        G_modified   : Tensor[N, d]  per-task modified gradients (Eq 2)
    """
    N = len(gradients)
    device = gradients[0].device
    G = torch.stack([g.float() for g in gradients], dim=0)  # [N, d] fp32

    # Pairwise cosine similarities  [N, N]
    G_norm = F.normalize(G, dim=1, eps=1e-8)
    cos_sim = G_norm @ G_norm.t()       # [N, N]

    # Eq 1 — projection weights  [N, N]
    # tau is differentiable; cos_sim may be detached (gradients treated as constants)
    w_proj = torch.sigmoid(k * (tau - cos_sim))     # [N, N]

    # Eq 2 — subtract conflicting projections
    # proj_{g_j}(g_i) = (g_i · g_j / ||g_j||^2) * g_j
    G_sq_norm = (G * G).sum(dim=1, keepdim=True).clamp(min=1e-12)   # [N, 1]
    dot_ij = G @ G.t()                               # [N, N]
    proj_coef = dot_ij / G_sq_norm.t()               # [N, N]  coef[i,j] = gi·gj/||gj||²

    # Weight by w_proj, zero diagonal (j != i)
    mask = 1.0 - torch.eye(N, device=device)
    weighted_coef = w_proj * proj_coef * mask        # [N, N]

    # g_i' = g_i - sum_j weighted_coef[i,j] * g_j
    G_modified = G - weighted_coef @ G               # [N, d]

    # Eq 3
    g_intervened = G_modified.sum(dim=0)             # [d]
    return g_intervened, G_modified


# ---------------------------------------------------------------------------
# Policy modules
# ---------------------------------------------------------------------------

class AIMScalarPolicy(nn.Module):
    """
    Scalar-policy AIM: single global threshold tau applied to every task pair.
    Parameter count: 1
    """

    def __init__(self, n_tasks: int, k: float = 10.0, init_tau: float = 0.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.k = k
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def get_tau_matrix(self) -> torch.Tensor:
        return self.tau.expand(self.n_tasks, self.n_tasks)

    def forward(
        self, gradients: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.get_tau_matrix()
        return aim_intervention(gradients, tau, self.k)


class AIMMatrixPolicy(nn.Module):
    """
    Matrix-policy AIM: unique threshold tau_ij per ordered task pair.
    Parameter count: N * N  (diagonal never used due to j != i mask)
    Provides the interpretable N x N heatmap from the paper.
    """

    def __init__(self, n_tasks: int, k: float = 10.0, init_tau: float = 0.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.k = k
        self.tau = nn.Parameter(torch.full((n_tasks, n_tasks), init_tau))

    def get_tau_matrix(self) -> torch.Tensor:
        return self.tau

    def forward(
        self, gradients: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return aim_intervention(gradients, self.tau, self.k)


# ---------------------------------------------------------------------------
# Policy loss (Equation 4)
# ---------------------------------------------------------------------------

class AIMPolicyLoss(nn.Module):
    """
    L_policy = lambda_g * L_guide + lambda_m * L_magnitude + lambda_p * L_progress

    Gradient flow:
      - L_magnitude and L_progress are differentiable w.r.t. tau through
        g_intervened = f(tau, g_i_detached).
      - L_guide = sum_i L_i(theta) does NOT depend on tau directly;
        it contributes alpha_i task weights used by L_progress.

    Hyperparameter ranges from paper Table 3 (Appendix A.2):
      lambda_g in [0, 1]
      lambda_m in [0, 0.01]
      lambda_p in [0, 0.08, 0.8]
      k = 10
    """

    def __init__(
        self,
        lambda_g: float = 1.0,
        lambda_m: float = 0.01,
        lambda_p: float = 0.08,
    ):
        super().__init__()
        self.lambda_g = lambda_g
        self.lambda_m = lambda_m
        self.lambda_p = lambda_p

    def forward(
        self,
        task_losses: List[torch.Tensor],
        gradients: List[torch.Tensor],
        g_intervened: torch.Tensor,
        guide_losses: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            task_losses  : per-task losses on guidance batch   [N scalars]
            gradients    : per-task flattened gradients (detached)  [N tensors of dim d]
            g_intervened : AIM-modified combined gradient  [d]   (differentiable w.r.t. tau)
            guide_losses : if provided, used as L_guide; else task_losses are used

        Returns:
            L_policy : scalar loss for policy optimizer
            info     : dict of individual loss component values
        """
        N = len(gradients)
        G = torch.stack(list(gradients), dim=0)   # [N, d]  fp32

        # --- Eq 5: Magnitude preservation ---
        g_norm = g_intervened.norm()
        sum_norms = sum(g.norm() for g in gradients)
        L_magnitude = (g_norm - sum_norms) ** 2

        # --- Eq 6: Progress on harder tasks ---
        loss_vals = torch.stack([l.detach() for l in task_losses])  # [N]
        alpha = loss_vals / (loss_vals.sum() + 1e-8)                # [N]  task weights
        # Dot product of combined gradient with each task gradient
        p_i = (g_intervened.unsqueeze(0) * G).sum(dim=1)           # [N]
        L_progress = -(alpha * p_i).sum()

        # --- L_guide: sum of task losses on guidance data ---
        ref_losses = guide_losses if guide_losses is not None else task_losses
        L_guide = torch.stack(ref_losses).sum()

        L_policy = (
            self.lambda_g * L_guide
            + self.lambda_m * L_magnitude
            + self.lambda_p * L_progress
        )

        info = {
            "L_policy":    L_policy.item(),
            "L_guide":     L_guide.item(),
            "L_magnitude": L_magnitude.item(),
            "L_progress":  L_progress.item(),
        }
        return L_policy, info


# ---------------------------------------------------------------------------
# Helper: flatten / unflatten parameter gradients
# ---------------------------------------------------------------------------

def flatten_grads(
    grads: Tuple[Optional[torch.Tensor], ...],
    params: List[torch.nn.Parameter],
) -> torch.Tensor:
    """Flatten a tuple of per-parameter gradients into a single vector."""
    parts = []
    for g, p in zip(grads, params):
        if g is not None:
            parts.append(g.flatten())
        else:
            parts.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(parts)


def set_grad_from_flat(
    flat_grad: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> None:
    """Write a flat gradient vector back into param.grad for each parameter."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.grad = flat_grad[offset: offset + numel].view(p.shape).clone()
        offset += numel
