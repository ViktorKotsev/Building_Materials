import argparse
from typing import Union, Tuple
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, SequentialSampler
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.iou_score import iou_loss
from utils.focal_loss import focal_loss

from unet_3.models.UNet_3Plus import UNet_3Plus

train_img = Path('/storage/group/building_materials/raw/CONFOCAL_SAMPLES/')
train_mask = Path('/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/ensemble_h_l_99')
val_img = Path('/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/domain_adaptation/vk4/general')
val_mask = Path('/usr/prakt/s0030/viktorkotsev_building_materials_ss2024/Unet/data/masks/10X_clean/Generalization_set')
dir_checkpoint = Path('./checkpoints/New/99_color')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_size: int = 256,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 0.5 Set a fixed seed for reproducibility
    # np.random.seed(42)

    # 1. Create dataset
    dataset_train = BasicDataset(train_img, train_mask, img_size, mode = 'train')
    dataset_val = BasicDataset(val_img, val_mask, img_size, mode = 'val')

    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset_train) * val_percent)
    # n_train = len(dataset_train) - n_val
    # #train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # # 2. Split into train / validation partitions
    # # obtain training indices that will be used for validation
    # num_train = len(dataset_train)
    # indices = list(range(num_train))
    # np.random.shuffle(indices)
    # split = int(np.floor((val_percent) * num_train))
    # train_idx, val_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SequentialSampler(val_idx)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(dataset_train, shuffle = True, **loader_args)
    val_loader = DataLoader(dataset_val, shuffle = False, drop_last=True, **loader_args)

    # (Initialize logging)
    n_train = len(dataset_train)
    n_val = len(dataset_val)
    experiment = wandb.init(project='U-Net', entity  ='building_materials_viktor', name = "Run")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_size=img_size, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:     {img_size}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8, foreach = True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, pct_start = 0.01, 
                                    total_steps= (epochs * n_train // batch_size) + 500, final_div_factor = 1e3)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    global_index = 0
    class_weights = torch.tensor([0.4782, 0.2792, 2.2426])

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # masks_pred_msssim = model(images)[1:]
                    if model.n_classes == 1:
                        #loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss = focal_loss(masks_pred.squeeze(1), true_masks, alpha=0.25, gamma=2, reduction='mean')
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        #loss = criterion(masks_pred, true_masks.squeeze(1))
                        loss = focal_loss(masks_pred, true_masks, alpha=class_weights, gamma=3, reduction='mean')
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks.squeeze(1), model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                scheduler.step()

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_dice, val_IoU = evaluate(model, val_loader, device, amp, global_index)
                        global_index = (global_index + 1) % (len(val_loader) * batch_size)
                        #scheduler.step(val_score)
                        # Used for wandb imgs
                        if model.n_classes == 1:
                            pred = torch.sigmoid(masks_pred) > 0.5
                        else:
                            pred = masks_pred.argmax(dim=1)

                        logging.info('Validation Dice score: {}'.format(val_dice))
                        logging.info('Validation IoU score: {}'.format(val_IoU))
                        experiment.log({
                            'validation Dice': val_dice,
                            'validation IoU': val_IoU,
                            'images': { 
                                'img': wandb.Image(images[0,:3,:,:].cpu()),
                                #'height': wandb.Image(images[0,3,:,:].cpu()),
                                #'light': wandb.Image(images[0,4,:,:].cpu()),
                            },
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(pred[0].float().cpu()),
                            },
                            # **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

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
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--size', '-s', type=parse_size, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    #model = UNet_3Plus(in_channels=5, n_classes = args.classes)
    model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Network:\n'
                 f'\t{total_params} parameters\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 #f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
                 )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.size,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.size,
            val_percent=args.val / 100,
            amp=args.amp
        )
