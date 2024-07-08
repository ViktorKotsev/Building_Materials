import torch
from torch import Tensor
import torch.nn.functional as F

def focal_loss(input: Tensor, target: Tensor, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> Tensor:
    """
    Compute the focal loss between `input` and `target`.

    Args:
        input (Tensor): Input tensor of shape (N, C, H, W) where C is the number of classes.
        target (Tensor): Target tensor of shape (N, H, W) with class indices.
        alpha (float): Weighting factor for the class.
        gamma (float): Focusing parameter.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Tensor: Computed focal loss.
    """
    # Convert target to one-hot encoding if needed
    if target.dim() == 3:
        target = F.one_hot(target, num_classes=input.size(1)).permute(0, 3, 1, 2).float()

    # Compute the log softmax of the input
    logpt = F.log_softmax(input, dim=1)
    pt = torch.exp(logpt)
    
    # Gather the log probabilities and probabilities corresponding to the target class
    logpt = logpt.gather(1, target.argmax(dim=1, keepdim=True)).squeeze(1)
    pt = pt.gather(1, target.argmax(dim=1, keepdim=True)).squeeze(1)

    # Apply alpha if it's a tensor, otherwise broadcast it
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.gather(0, target.argmax(dim=1).flatten()).view_as(pt)
    else:
        alpha = torch.tensor(alpha).to(input.device)
    
    # Compute the focal loss
    focal_loss = -alpha * (1 - pt) ** gamma * logpt

    # Apply reduction method
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

