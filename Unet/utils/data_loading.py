import logging
import numpy as np
import torch
from PIL import Image, ImageEnhance
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
import torch.nn.functional as F

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

def load_height(filename):
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

    height_img = Image.fromarray(height_matrix)  
    return height_img


def load_light(filename):
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

    # Concatenate the height tensor with the torch_img tensor
    light_img = Image.fromarray(light_matrix)    
    return light_img


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, size: 256, mask_suffix: str = '', mode = 'train', height=False, light=False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)

        self.size = size
        self.mode = mode
        self.height = height
        self.light = light
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
    def preprocess(pil_img, size, is_mask):
        newH, newW = size
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        # Convert airVoids to white and aggregates to gray
        if is_mask:
            mask = img.copy()
            mask[(mask > 0) & (mask <= 128)] = 1
            mask[mask > 128] = 2
            # mask[mask == 128] = 1
            # mask[mask == 255] = 2

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
        
    @staticmethod    
    def preprocess_additional(img, size):
        img_matrix = np.array(img)
        img_matrix = (img_matrix - img_matrix.min() ) / (img_matrix.max() - img_matrix.min()) *255.0
        img_new = Image.fromarray(img_matrix)

        newH, newW = size
        img_new = img_new.resize((newW, newH), Image.BICUBIC)
        img_matrix = np.asarray(img_new)
        # Convert the height matrix to a PyTorch tensor
        img_tensor = torch.tensor(img_matrix, dtype=torch.float32).unsqueeze(0)

        # Normalize between 0-1
        img_tensor = img_tensor/ 255.0
        
        return img_tensor

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.rglob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.rglob(name + '.*'))
        # height_file = list(self.images_dir.rglob(name + '.vk4'))
        # light_file = list(self.images_dir.rglob(name + '.vk4'))

        #assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # Crop 20 pixels to fit the masks
        if self.mode == 'val':
            img = img.crop((20, 20, img.size[0] - 20, img.size[1] - 20))
        #mask = mask.crop((20, 20, mask.size[0] - 20, mask.size[1] - 20))

        if self.height:
            height_img = load_height(img_file[0])
            if self.mode == 'val':
                height_img = height_img.crop((20, 20, height_img.size[0] - 20, height_img.size[1] - 20))

        if self.light:
            light_img = load_light(img_file[0])
            light_img = light_img.crop((20, 20, light_img.size[0] - 20, light_img.size[1] - 20))

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        
        if self.mode == 'train':
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                if self.height:
                    height_img = height_img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.light:
                    light_img = light_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Data augmentation: Random vertical flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                if self.height:
                    height_img = height_img.transpose(Image.FLIP_TOP_BOTTOM)
                if self.light:
                    light_img = light_img.transpose(Image.FLIP_TOP_BOTTOM)

            # # Data augmentation: Random crop with varying sizes between 0.5 and 1.0 of the original image
            # scale = random.uniform(0.5, 1.0)
            # width, height = img.size
            # new_width = int(scale * width)
            # new_height = int(scale * height)

            # # Ensure the crop size is not larger than the image size
            # if new_width < width or new_height < height:
            #     # Randomly select the top-left corner for cropping
            #     left = random.randint(0, width - new_width)
            #     top = random.randint(0, height - new_height)
            #     right = left + new_width
            #     bottom = top + new_height

            #     # Crop both image and mask
            #     img = img.crop((left, top, right, bottom))
            #     mask = mask.crop((left, top, right, bottom))
            #     if self.height:
            #         height_img = height_img.crop((left, top, right, bottom))
            #     if self.light:
            #         light_img = light_img.crop((left, top, right, bottom))
                
            
            # Adjust brightness
            if random.random() < 0.1:
                enhancer = ImageEnhance.Brightness(img)
                factor = random.uniform(0.95, 1.05)
                img = enhancer.enhance(factor)

            # Adjust contrast
            if random.random() < 0.1:
                enhancer = ImageEnhance.Contrast(img)
                factor = random.uniform(0.95, 1.05)
                img = enhancer.enhance(factor)

            # Adjust color (saturation)
            if random.random() < 0.1:
                enhancer = ImageEnhance.Color(img)
                factor = random.uniform(0.95, 1.05)
                img = enhancer.enhance(factor)

            # Adjust sharpness
            if random.random() < 0.1:
                enhancer = ImageEnhance.Sharpness(img)
                factor = random.uniform(0.95, 1.05)
                img = enhancer.enhance(factor)


        img = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)
        
        img = torch.as_tensor(img).float()
        mask = torch.as_tensor(mask).long()

        # Normalize and add additional information
        if self.height:
            height_tensor = self.preprocess_additional(height_img, self.size)
            img = torch.cat((img, height_tensor), dim=0)

        if self.light:
            light_tensor = self.preprocess_additional(light_img, self.size)
            img = torch.cat((img, light_tensor), dim=0)


        data = {
            'image': img.float().contiguous(),
            'mask': mask.contiguous()
        }

        return data