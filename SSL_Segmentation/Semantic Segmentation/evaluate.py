from myutil import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Evaluate_metric import dice_coeff
# from Data_prepare import MyDataset
# from torch.utils.data import DataLoader


def evaluate(model, val_loader, device):
    model.eval()
    dice_score = 0
    num_val_batches = len(val_loader)
    with torch.no_grad():
        for i, (images, true_masks) in enumerate(val_loader):
            images = images.to(torch.float32).to(device=device)
            masks_pred = model(images)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            dice_score += dice_coeff(masks_pred, true_masks.float()).to(device).item()

    return dice_score / max(num_val_batches, 1)


# val_img_path = "Data/val/thumb"
# val_mask_path = "Data/val/mask"
# # train_set = MyDataset(train_img_path, train_mask_path, 256)
# val_set = MyDataset(val_img_path, val_mask_path, 512)
# # train_loader = DataLoader(train_set, batch_size=6, shuffle=False)
# val_loader = DataLoader(val_set, batch_size=6, shuffle=False)
#
# for i, (images, true_masks) in enumerate(val_loader):
#     print(images.shape, true_masks.shape)