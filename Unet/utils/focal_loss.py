import torch
from torch import Tensor
import torch.nn.functional as F

def focal_loss(input: Tensor, target: Tensor, alpha=None, gamma: float = 2.0, reduction: str = 'mean') -> Tensor:
    """
    Compute the focal loss between `input` and `target`, addressing class imbalance.

    Args:
        input (Tensor): Input tensor of shape (N, C, H, W), logits before softmax.
        target (Tensor): Target tensor of shape (N, H, W) with class indices in [0, C-1].
        alpha (Tensor or float, optional): Weighting factor for classes. Should be a Tensor of size (C,) or a scalar.
        gamma (float): Focusing parameter.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Tensor: Computed focal loss.
    """
    # Ensure input and target are on the same device
    device = input.device
    target = target.to(device)

    # Reshape input and target tensors to (N*H*W, C) and (N*H*W)
    N, C, H, W = input.shape
    input_flat = input.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.view(-1)

    # Compute log probabilities
    logpt = F.log_softmax(input_flat, dim=1)  # (N*H*W, C)
    pt = torch.exp(logpt)  # (N*H*W, C)

    # Select the log probabilities and probabilities corresponding to the target class
    logpt_target = logpt[range(target_flat.size(0)), target_flat]  # (N*H*W,)
    pt_target = pt[range(target_flat.size(0)), target_flat]        # (N*H*W,)

    # Apply alpha (class weights)
    if alpha is not None:
        if isinstance(alpha, Tensor):
            if alpha.device != device:
                alpha = alpha.to(device)
            alpha_t = alpha[target_flat]  # Gather alpha values for each target
        else:
            alpha_t = torch.full_like(target_flat, fill_value=alpha, device=device)
    else:
        alpha_t = torch.ones_like(target_flat, device=device)

    # Compute focal loss
    loss = -alpha_t * (1 - pt_target) ** gamma * logpt_target  # (N*H*W,)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.view(N, H, W)  # Return per-pixel loss
