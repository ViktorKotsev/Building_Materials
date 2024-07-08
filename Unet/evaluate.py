import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import random

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou_score import iou_score


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, global_index):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score_total = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        cumulative_image_count = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image).squeeze(dim=1)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # compute the IOU score
                iou_score_total += iou_score(mask_pred, mask_true, 2)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true_hot = F.one_hot(mask_true.squeeze(1), net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred_hot[:, 1:], mask_true_hot[:, 1:], reduce_batch_first=False)
                # compute the IOU score
                iou_score_total += iou_score(mask_pred.argmax(dim=1), mask_true.squeeze(1), net.n_classes)

            # Calculate batch start and end indices
            batch_start_idx = cumulative_image_count
            batch_end_idx = cumulative_image_count + len(image)

            # Log an image to wandb (only once per evaluation)
            if batch_start_idx <= global_index < batch_end_idx:
                local_idx = global_index - batch_start_idx
                img = image[local_idx].cpu().numpy().transpose((1, 2, 0))
                mask_t = mask_true[local_idx].cpu().squeeze().numpy()
                mask_t = mask_t * 120
                if net.n_classes == 1:
                    mask_p = mask_pred[local_idx].cpu().squeeze().numpy()
                else:
                    mask_p = mask_pred[local_idx].cpu().argmax(dim=0).numpy()
                    mask_p = mask_p * 120
                wandb.log({
                    f"val/image": wandb.Image(img[:,:,:3], caption="Input Image"),
                    # f"val/height": wandb.Image(img[:,:,3], caption="Input Image"),
                    # f"val/light": wandb.Image(img[:,:,4], caption="Input Image"),
                    f"val/mask_true": wandb.Image(mask_t, caption="True Mask"),
                    f"val/mask_pred": wandb.Image(mask_p, caption="Predicted Mask")
                })
            cumulative_image_count = batch_end_idx

    net.train()
    return dice_score / max(num_val_batches, 1), iou_score_total / max(num_val_batches, 1)
