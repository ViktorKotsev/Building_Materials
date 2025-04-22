import os
import numpy as np
from PIL import Image
import torch

def compute_class_weights_from_masks(mask_folder, num_classes):
    """
    Computes class weights from mask images in a folder.

    Args:
        mask_folder (str): Path to the folder containing mask PNG images.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Class weights tensor of shape (num_classes,).
    """
    class_counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0

    # List all PNG files in the folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        # Open the mask image
        with Image.open(mask_path) as mask:
            # Convert mask to numpy array
            mask_np = np.array(mask)
            mask_np[(mask_np > 0) & (mask_np <= 128)] = 1
            mask_np[mask_np > 128] = 2
            # Ensure mask is a 2D array
            if mask_np.ndim == 3:
                # If mask is RGB, convert to grayscale by taking the first channel
                mask_np = mask_np[..., 0]
            elif mask_np.ndim != 2:
                raise ValueError(f"Unexpected mask dimensions in file {mask_file}")

            # Flatten the mask and count occurrences of each class
            mask_flat = mask_np.flatten()
            for cls in range(num_classes):
                class_counts[cls] += np.sum(mask_flat == cls)
            total_pixels += mask_flat.size

    # Avoid division by zero for classes not present in the dataset
    class_counts = np.maximum(class_counts, 1e-6)

    # Compute class weights inversely proportional to class frequencies
    class_weights = total_pixels / (num_classes * class_counts)
    # Normalize the weights to sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes

    # Convert to torch Tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights


mask_folder = '/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/10X_clean/Training_set'
num_classes = 3  # Replace with the actual number of classes

# Compute class weights
class_weights = compute_class_weights_from_masks(mask_folder, num_classes)

print('Class Weights:', class_weights)