import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from baal.bayesian.dropout import MCDropoutModule

from utils.data_loading import BasicDataset, add_heightMap, load_image, add_lightMap
from unet import UNet
from utils.utils import plot_img_and_mask, plot_uncertainty

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                confidence_threshold=0.95,
                mc_iterations=10,
                img_data = None,
                filename = None):
    net.eval()
    img = BasicDataset.preprocess(full_img, scale_factor, is_mask=False)

    img = torch.from_numpy(img)

    # img = add_heightMap(img_data, img, scale_factor)
    # img = add_lightMap(img_data, img, scale_factor)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask.long().squeeze().numpy()


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
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
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

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder[0]):

        input_path = os.path.join(input_folder[0], filename)
        logging.info(f'Predicting image {input_path} ...')
            
        img = load_image(input_path)

        if not img:
            continue

        mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device,
                            img_data=input_path,
                            filename=filename)           

        if not args.no_save:
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    process_folder(in_files, out_files, net, device, args)