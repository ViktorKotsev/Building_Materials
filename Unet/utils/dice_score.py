import torch
import torch.nn.functional as F
from torch import Tensor

def dice_loss(
    pred: Tensor,
    target: Tensor,
    n_classes: int = 3,
    ignore_index: int = 100,
    epsilon: float = 1e-6
) -> Tensor:
    """
    Computes the average multi-class Dice score, ignoring pixels where target == ignore_index.

    Args:
        pred (Tensor): Model predictions of shape (N, C, H, W), 
                       typically probabilities or logits for each class.
        target (Tensor): Ground-truth labels of shape (N, H, W) or (N, 1, H, W).
                         Values are class indices in [0, C-1], except for `ignore_index`.
        n_classes (int): Number of classes.
        ignore_index (int): Label to ignore in the Dice calculation.
        epsilon (float): Smoothing constant to avoid division by zero.

    Returns:
        Tensor: Dice Loss
    """
    # 1) If target is (N, H, W), unsqueeze to (N, 1, H, W) so it can broadcast with (N, C, H, W).
    if target.dim() == 3:
        target = target.unsqueeze(1)  # => (N, 1, H, W)

    # 2) Create a boolean mask of valid pixels: (target != ignore_index).
    #    Shape: (N, 1, H, W), broadcasting-compatible with pred.
    valid_mask = (target != ignore_index)

    # 3) Zero out invalid pixels in both 'pred' and 'target' by multiplying with valid_mask.
    #    If your predictions are raw logits, you might want to convert them to probabilities:
    #    pred = torch.softmax(pred, dim=1)  # <= Uncomment if 'pred' are raw logits
    pred_valid = pred * valid_mask  # (N, C, H, W)
    target_valid = target * valid_mask  # (N, 1, H, W)

    # 4) Convert the integer target labels to one-hot format of shape (N, C, H, W).
    #    Squeeze out the channel dimension before one_hot, then permute back.
    #    NOTE: only valid pixels remain non-zero, because invalid pixels are zeroed out above.
    target_onehot = F.one_hot(
        target_valid.squeeze(1).long(),  # one_hot requires long type
        num_classes=n_classes
    ).permute(0, 3, 1, 2).float()  # => (N, C, H, W)

    # 5) Compute Dice score per class:
    #    intersection_c = ∑(pred_c * target_c)
    #    union_c       = ∑(pred_c) + ∑(target_c)
    #    dice_c        = (2 * intersection_c + eps) / (union_c + eps)
    #    Then average over classes.
    intersection = 2.0 * (pred_valid * target_onehot).sum(dim=(0, 2, 3))  # sum over N,H,W
    union = pred_valid.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

    dice_per_class = (intersection + epsilon) / (union + epsilon)

    # 6) Loss == 1 - Dice score mean
    return 1 - dice_per_class.mean()