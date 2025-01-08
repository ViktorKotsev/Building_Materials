import argparse
import logging
import os
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset, load_height, load_image, load_light
from unet import UNet
from utils.utils import plot_img_and_mask, plot_uncertainty
from unet_3.models.UNet_3Plus import UNet_3Plus_DeepSup_CGM, UNet_3Plus

def predict_img(net,
                full_img,
                device,
                size=512,
                out_threshold=0.5,
                confidence_threshold=0.95,
                mc_iterations=10,
                img_data = None,
                filename = None):
    net.eval()
    img = BasicDataset.preprocess(full_img, size, is_mask=False)
    img = torch.as_tensor(img.copy()).float()
    # img = torch.from_numpy(img)

    # # Add height data
    # height_img = load_height(img_data)
    # height_tensor = BasicDataset.preprocess_additional(height_img, size)
    # img = torch.cat((img, height_tensor), dim=0)

    # # Add light data
    # light_img = load_light(img_data)
    # light_tensor = BasicDataset.preprocess_additional(light_img, size)
    # img = torch.cat((img, light_tensor), dim=0)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bicubic')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask.long().squeeze().numpy()

def parse_size(value: str) -> Union[int, Tuple[int, int]]:
    """
    Parse the `--size` argument to allow an integer or a tuple (width, height).

    """
    try:
        if ',' in value:  # If the value contains a comma, treat it as a tuple
            width, height = map(int, value.split(','))
            return width, height
        else:  # Otherwise, parse it as a single integer
            return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Size must be an int or a tuple of two ints, e.g., '256' or '256,128'.")
    
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch120.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--size', '-s', type=parse_size, default=256,
                        help='Size for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    out = out * 120
    return Image.fromarray(out)

def process_folder(input_folder, output_folder, net, device, args):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder[0], exist_ok=True)
    
    net.eval()

    # Iterate over all files in the input folder and its subfolders
    for root, dirs, files in os.walk(input_folder[0]):
        for filename in files:
            input_path = os.path.join(root, filename)
            logging.info(f'Predicting image {input_path} ...')
                
            img = load_image(input_path)

            if not img:
                continue

            mask = predict_img(net=net,
                                full_img=img,
                                size=args.size,
                                out_threshold=args.mask_threshold,
                                device=device,
                                img_data=input_path,
                                filename=filename)           

            if not args.no_save:
                # Save output file directly to the main output folder
                out_filename = os.path.join(output_folder[0], f'{os.path.splitext(filename)[0]}.png')
                result = mask_to_image(mask, [0, 1, 2])
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)        
        
        

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = args.output

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = UNet_3Plus(in_channels=5, n_classes = args.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    process_folder(in_files, out_files, net, device, args)