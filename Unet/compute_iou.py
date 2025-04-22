import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from utils.iou_score import iou_score
from utils.data_loading import BasicDataset

def compute_multiclass_iou(mask_dir, pred_dir, num_classes, threshold=0.5):
    mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
    pred_files = [f for f in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, f))]
    pred_files = [f for f in pred_files if f in mask_files]

    assert len(mask_files) == len(pred_files), "The number of mask and prediction files should be the same."

    iou_scores = []

    for mask_file, pred_file in zip(mask_files, pred_files):
        mask_path = os.path.join(mask_dir, mask_file)
        pred_path = os.path.join(pred_dir, pred_file)

        
        mask_true = Image.open(mask_path)
        mask_pred = Image.open(pred_path)

        # Cut top and bottom 20 pixels since the ground truth has a bug
        width, height = mask_pred.size
        mask_pred = mask_pred.crop((20, 20, width - 20, height - 20))

        # Adjust masks
        mask_true = np.array(mask_true)
        mask_true[(mask_true > 0) & (mask_true <= 128)] = 1
        mask_true[mask_true > 128] = 2

        mask_pred = np.array(mask_pred)
        mask_pred[(mask_pred > 0) & (mask_pred <= 128)] = 1
        mask_pred[mask_pred > 128] = 2

        mask_true = torch.tensor(mask_true, dtype=torch.long)
        mask_pred = torch.tensor(mask_pred, dtype=torch.long)

        iou = iou_score(mask_pred, mask_true, num_classes)
        iou_scores.append(iou[1])

    mean_iou = np.mean(iou_scores)
    return mean_iou

# Example usage:
mask_directory = "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/domain_adaptation/general"
prediction_directory = "/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/eval_4/pseudo/97_h_color_2/general"
num_classes = 3  # Adjust this number based on the number of classes in your masks
mean_iou = compute_multiclass_iou(mask_directory, prediction_directory, num_classes)
print(f"Mean IoU: ", mean_iou)