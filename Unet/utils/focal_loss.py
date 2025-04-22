import torch
import torch.nn.functional as F
from torch import Tensor

def focal_loss(
    input: Tensor,
    target: Tensor,
    alpha=None,
    gamma: float = 2.0,
    ignore_index: int = 100,
    reduction: str = 'mean'
) -> Tensor:
    """
    Compute the focal loss between `input` and `target`, ignoring pixels where target == ignore_index.
    
    Args:
        input (Tensor): (N, C, H, W), logits before softmax.
        target (Tensor): (N, H, W) with class indices in [0, C-1], or == ignore_index for ignored pixels.
        alpha (Tensor or float, optional): Class weighting factor.
        gamma (float): Focusing parameter.
        ignore_index (int): Label to ignore in the loss.
        reduction (str): 'none' | 'mean' | 'sum'.
    
    Returns:
        Tensor: Computed focal loss (scalar by default).
    """
    device = input.device
    target = target.to(device)

    # Flatten predictions and targets
    N, C, H, W = input.shape
    input_flat = input.permute(0, 2, 3, 1).reshape(-1, C)  # shape: (N*H*W, C)
    target_flat = target.view(-1)                         # shape: (N*H*W,)

    # -------------------------------------------------------------------------
    # 1) Create mask for valid pixels (those != ignore_index)
    # -------------------------------------------------------------------------
    valid_mask = (target_flat != ignore_index)
    # If nothing is valid, return zero loss
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Keep only valid pixels
    input_flat = input_flat[valid_mask]
    target_flat = target_flat[valid_mask]

    # Compute log probabilities for valid pixels
    logpt = F.log_softmax(input_flat, dim=1)   # (valid_count, C)
    pt = logpt.exp()                           # (valid_count, C)

    # Select the log probabilities and probabilities corresponding to the target class
    idx = torch.arange(logpt.size(0), device=device)
    logpt_target = logpt[idx, target_flat]  # (valid_count,)
    pt_target   = pt[idx, target_flat]      # (valid_count,)

    # -------------------------------------------------------------------------
    # 2) Apply class weighting alpha, if provided
    # -------------------------------------------------------------------------
    if alpha is not None:
        if isinstance(alpha, Tensor):
            if alpha.device != device:
                alpha = alpha.to(device)                 # shape: (C,)
            alpha_t = alpha[target_flat]            # gather per-pixel alpha
        else:
            alpha_t = torch.full_like(target_flat, fill_value=alpha, device=device)
    else:
        alpha_t = torch.ones_like(target_flat, device=device)

    # -------------------------------------------------------------------------
    # 3) Compute focal loss
    # -------------------------------------------------------------------------
    focal_term = (1.0 - pt_target) ** gamma
    loss = -alpha_t * focal_term * logpt_target  # (valid_count,)

    # -------------------------------------------------------------------------
    # 4) Reduction
    # -------------------------------------------------------------------------
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        # 'none': return per-pixel (valid) loss in original shape
        # We need to un-flatten it back to (N,H,W), with ignored pixels = 0
        out = torch.zeros(N*H*W, device=device, dtype=loss.dtype)
        out[valid_mask] = loss
        return out.view(N, H, W)
