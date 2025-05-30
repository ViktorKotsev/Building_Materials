import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import random

from utils.iou_score import iou_score
from torchvision.transforms import Resize


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, global_index):
    net.eval()
    num_val_batches = len(dataloader)
    iou_score_total = 0
    iou_per_class = torch.zeros(net.n_classes, device='cuda' if torch.cuda.is_available() else 'cpu')

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

                # compute the IOU score
                iou_score_total += iou_score(mask_pred, mask_true, 2)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
        
                # compute the IOU score (both mean and per class)
                batch_iou_score, batch_iou_per_class = iou_score(mask_pred.argmax(dim=1), mask_true.squeeze(1), net.n_classes)

                iou_score_total += batch_iou_score
                iou_class_tensor = torch.tensor(batch_iou_per_class, device=iou_per_class.device)
                iou_per_class += iou_class_tensor

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

    avg_iou_per_class = [score / max(num_val_batches, 1) for score in iou_per_class]
    wandb.log({
         **{f"iou_class_{i}": iou.item() for i, iou in enumerate(avg_iou_per_class)}
    })

    net.train()
    return iou_score_total / max(num_val_batches, 1)
