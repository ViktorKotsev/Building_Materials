import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def iou_score(pred, target, num_classes):
    iou = 0
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            iou += torch.tensor(1.0)  # If there is no ground truth and prediction for this class
        else:
            iou += intersection / union
    return iou / num_classes



def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    # Average of IoU coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter
    union = torch.where(union == 0, union + epsilon, union)

    iou = (inter + epsilon) / union
    return iou.mean()

def multiclass_iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    # Average of IoU coefficient for all classes
    return iou_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def iou_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    # IoU loss (objective to minimize) between 0 and 1
    fn = multiclass_iou_coeff if multiclass else iou_coeff
    return 1 - fn(input, target, reduce_batch_first=True)