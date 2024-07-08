import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import random
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

import vk4extract


def load_image(filename):
    try:
        ext = splitext(filename)[1]
        if ext == '.npy':
            rgb = np.load(filename)
            # Normalize
            min_across_channels = np.expand_dims(np.expand_dims(rgb.min(0).min(0),0),0)
            max_across_channels = np.expand_dims(np.expand_dims(rgb.max(0).max(0),0),0)
            normalized_rgb= (rgb - min_across_channels) / (max_across_channels - min_across_channels)
            return Image.fromarray(normalized_rgb)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext == '.vk4': 
            with open(filename, 'rb') as in_file:
                offsets = vk4extract.extract_offsets(in_file)
                rgb_dict = vk4extract.extract_color_data(offsets, 'peak', in_file)
            rgb_data = rgb_dict['data']
            height = rgb_dict['height']
            width = rgb_dict['width']
            image_array = np.reshape(rgb_data, (height, width, 3))[...,::-1]  # Convert from BGR to RGB

            return Image.fromarray(image_array)        
        else:
            return Image.open(filename)
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        return None

def add_heightMap(filename, torch_img, scale):
    ext = splitext(filename)[1]
    if ext == '.vk4':
        with open(filename, 'rb') as in_file:
            offsets = vk4extract.extract_offsets(in_file)
            height_dict = vk4extract.extract_img_data(offsets, 'height', in_file)

        height_data = height_dict['data']
        height = height_dict['height']
        width = height_dict['width']
        height_matrix = np.reshape(height_data, (height, width))
    elif ext == '.npy':
        height_matrix = np.load(filename)
        
    # Normalize
    height_matrix = (height_matrix - height_matrix.min() ) / (height_matrix.max() - height_matrix.min()) *255.0

    if scale != 1:
        height_img = Image.fromarray(height_matrix)
        w, h = height_img.size
        newW, newH = int(scale * w), int(scale * h)
        height_img = height_img.resize((newW, newH), Image.BICUBIC)
        height_matrix = np.asarray(height_img)

    # Convert the height matrix to a PyTorch tensor
    height_tensor = torch.tensor(height_matrix, dtype=torch.float32).unsqueeze(0)

    # Normalize between 0-1
    height_tensor = height_tensor/ 255.0

    # Concatenate the height tensor with the torch_img tensor
    torch_img = torch.cat((torch_img, height_tensor), dim=0)    
    return torch_img

def add_lightMap(filename, torch_img, scale):
    ext = splitext(filename)[1]
    if ext == '.vk4':
        with open(filename, 'rb') as in_file:
            offsets = vk4extract.extract_offsets(in_file)
            light_dict = vk4extract.extract_img_data(offsets, 'light', in_file)

        light_data = light_dict['data']
        height = light_dict['height']
        width = light_dict['width']
        light_matrix = np.reshape(light_data, (height, width))
    elif ext == '.npy':
        light_matrix = np.load(filename)
    # Normalize
    light_matrix = (light_matrix - light_matrix.min() ) / (light_matrix.max() - light_matrix.min()) *255.0

    if scale != 1:
        light_img = Image.fromarray(light_matrix)
        w, h = light_img.size
        newW, newH = int(scale * w), int(scale * h)
        light_img = light_img.resize((newW, newH), Image.BICUBIC)
        light_matrix = np.asarray(light_img)
    
    # Convert the light matrix to a PyTorch tensor
    light_tensor = torch.tensor(light_matrix, dtype=torch.float32).unsqueeze(0)

    # Normalize between 0-1
    light_tensor = light_tensor/ 255.0

    # Concatenate the height tensor with the torch_img tensor
    torch_img = torch.cat((torch_img, light_tensor), dim=0)    
    return torch_img


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', mode = 'train'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mode = mode
        self.mask_suffix = mask_suffix
        

        # Recursively find all image files, excluding specified directories
        self.image_files = [file for file in self.images_dir.rglob('*.*') if isfile(file)]
        all_ids = [splitext(file.name)[0] for file in self.image_files if isfile(file)]
        
        # Collect all mask IDs
        all_mask_files = list(self.mask_dir.rglob(f'*{mask_suffix}.*'))
        mask_ids = {splitext(file.name)[0] for file in all_mask_files if isfile(file)}

        # Filter the ids to only include those for which masks exist,
        # and exclude redundant imgs
        seen_ids = set()
        self.ids = []
        for img_id in all_ids:
            if img_id in mask_ids and img_id not in seen_ids:
                self.ids.append(img_id)
                seen_ids.add(img_id)

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        # Convert airVoids to white and aggregates to gray
        if is_mask:
            mask = img.copy()
            mask[(mask > 0) & (mask <= 120)] = 1
            mask[mask > 120] = 2

            return mask    
        else:

            # Normalize
            min_across_channels = np.expand_dims(np.expand_dims(img.min(0).min(0),0),0)
            max_across_channels = np.expand_dims(np.expand_dims(img.max(0).max(0),0),0)
            img = (img - min_across_channels) / (max_across_channels - min_across_channels)

            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.rglob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.rglob(name + '.*'))
        # height_file = list(self.images_dir.rglob(name + '.vk4'))
        # light_file = list(self.images_dir.rglob(name + '.vk4'))

        #assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        img = torch.as_tensor(img.copy()).float()
        mask = torch.as_tensor(mask.copy()).long()

        # Augmentations
        # if self.mode == 'train':
        #     # Random Flips
        #     if random.random() > 0.25:
        #         img = F.hflip(img)
        #         mask = F.hflip(mask)
            
        #     if random.random() > 0.25:
        #         img = F.vflip(img)
        #         mask = F.vflip(mask)
            
        #     # Slight Color Changes
        #     if random.random() > 0.1:
        #         img = F.adjust_brightness(img, brightness_factor=random.uniform(0.9, 1.1))
        #     if random.random() > 0.1:
        #         img = F.adjust_contrast(img, contrast_factor=random.uniform(0.9, 1.1))
        #     if random.random() > 0.1:
        #         img = F.adjust_saturation(img, saturation_factor=random.uniform(0.9, 1.1))
        #     if random.random() > 0.1:
        #         img = F.adjust_hue(img, hue_factor=random.uniform(-0.05, 0.05))
            
        #     # Apply Gaussian Blur
        #     if random.random() > 0.1:
        #         kernel_size = random.choice([3, 5])  # Choosing between different kernel sizes for blur effect
        #         sigma = random.uniform(0.1, 2.0)  # Standard deviation for Gaussian kernel
        #         img = F.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

        # Add height Map
        img = add_heightMap(img_file[0], img, self.scale)
        img = add_lightMap(img_file[0], img, self.scale)       

        return {
            'image': img.float().contiguous(),
            'mask': mask.contiguous()
        }
