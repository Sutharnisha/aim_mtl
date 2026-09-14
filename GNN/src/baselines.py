"""
Baseline multi-task optimizers used in the AIM paper comparisons.

  LS      — Linear Scalarization (equal-weight sum of task losses)
  PCGrad  — Project Conflicting Gradients (Yu et al., NeurIPS 2020)

Both return a combined gradient vector with the same shape as the
per-task gradients, so they plug into the same training loop as AIM.
"""

import torch
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Linear Scalarization (LS)
# ---------------------------------------------------------------------------

def ls_combine_losses(task_losses: List[torch.Tensor]) -> torch.Tensor:
    """
    Simple equal-weight average of task losses.
    Calling .backward() on the result gives the LS combined gradient.
    """
    return torch.stack(task_losses).mean()


def ls_combine_gradients(flat_grads: List[torch.Tensor]) -> torch.Tensor:
    """LS: mean of the per-task gradients."""
    return torch.stack(flat_grads, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# PCGrad — Project Conflicting Gradients
# ---------------------------------------------------------------------------

def pcgrad_combine(flat_grads: List[torch.Tensor]) -> torch.Tensor:
    """
    PCGrad (Yu et al., 2020): for each task i, project out the component
    of g_i that conflicts with g_j (i.e., cos(g_i, g_j) < 0).

    Args:
        flat_grads : List[Tensor[d]]  per-task flattened gradients (detached)

    Returns:
        Tensor[d]  combined gradient
    """
    N = len(flat_grads)
    result = torch.zeros_like(flat_grads[0])

    for i in range(N):
        g_i = flat_grads[i].clone()

        for j in range(N):
            if i == j:
                continue
            dot     = (g_i * flat_grads[j]).sum()
            norm_sq = flat_grads[j].dot(flat_grads[j]).clamp(min=1e-12)
            cos_ij  = (dot / (g_i.norm() * flat_grads[j].norm().clamp(min=1e-8))).item()

            if cos_ij < 0.0:
                g_i.sub_((dot / norm_sq) * flat_grads[j])

        result.add_(g_i)

    # Average (not sum) so the encoder step matches LS's mean-gradient scale.
    return result / N


# ---------------------------------------------------------------------------
# Dispatcher — unified interface for training loop
# ---------------------------------------------------------------------------

BASELINE_METHODS = ("ls", "pcgrad")


def combine_gradients(
    method: str,
    flat_grads: List[torch.Tensor],
) -> torch.Tensor:
    """
    Unified dispatcher: returns combined gradient for given method.

    Args:
        method     : "ls" | "pcgrad"
        flat_grads : per-task flattened gradients (detached)
    """
    if method == "ls":
        return ls_combine_gradients(flat_grads)
    elif method == "pcgrad":
        return pcgrad_combine(flat_grads)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}")
