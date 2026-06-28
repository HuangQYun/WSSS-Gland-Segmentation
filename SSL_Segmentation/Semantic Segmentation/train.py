from myutil import *
from Data_prepare import MyDataset
from torch.utils.data import DataLoader
import time
from Model.unet_model import Unet
import argparse
import torch
from torch import optim
from Evaluate_metric import dice_loss
from tqdm import tqdm
from evaluate import evaluate
import logging
from os.path import join
from Model.Mymodel import Model
logger_fmt = '%(asctime)s - %(funcName)s - %(message)s'


def train(args):
    # print(device)
    train_logger = Log("train", 'train_log')
    train_img_path = "Data/train/thumb"
    train_mask_path = "Data/train/mask"
    val_img_path = "Data/val/thumb"
    val_mask_path = "Data/val/mask"
    train_set = MyDataset(train_img_path, train_mask_path, args.crop_size, mode="train")
    val_set = MyDataset(val_img_path, val_mask_path, args.crop_size, mode="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    model = Unet().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = torch.nn.BCELoss()
    epochs = args.epochs
    best_loss = 0
    epoch_logger = Log("epoch", 'epoch_log')
    val_logger = Log("val", 'val_log')
    print('len:', len(train_loader))
    for epoch in tqdm(range(epochs)):
        average_loss = 0
        interval = 0
        for images, true_masks in train_loader:
            model.train()
            images = images.to(torch.float32).to(device=device)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # print(f'images.shape:{images.shape}, true_masks.shape:{true_masks.shape}')
            # print(f'images.dtype:{images.dtype}, true_masks.dtype:{true_masks.dtype}')
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks.float())
            # print(masks_pred.shape, true_masks.shape)
            loss += dice_loss(masks_pred, true_masks.float()).to(device)
            train_logger.info(f'epoch:{epoch + 1}, train loss:{loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss += loss
            if (interval+1) % 50 == 0:
                val_loss = evaluate(model, val_loader, device)
                is_best = val_loss > best_loss
                val_logger.info(f'val_loss:{val_loss}')
                best_loss = val_loss if val_loss > best_loss else best_loss
                save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                 'optim_dict': optimizer.state_dict()},
                                epoch=epoch, is_best=is_best, load_path=args.load, sava_epoch=True)

            interval += 1
        epoch_logger.info(f'epoch:{epoch + 1}, average_loss:{average_loss / len(train_loader)}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    train(args)