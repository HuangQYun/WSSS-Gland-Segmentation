import time

from myutil import *
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import join, splitext, isfile
from collections import Counter
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, crop_size, transform_img=None, transform_mask=None, mode=""):
        self.crop_size = crop_size
        self.img_base_path = os.path.abspath(image_dir)
        self.mask_base_path = os.path.abspath(mask_dir)
        self.img_set = [file for file in os.listdir(image_dir) if '.png' in file]
        self.mask_set = [file for file in os.listdir(mask_dir) if '.png' in file]
        self.mode = mode
        if transform_img is None:
            self.trans_img = transforms.Compose([
                # transforms.Resize(400),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans_img = transform_img

        if transform_mask is None:
            self.trans_mask = transforms.Compose([
                # transforms.Resize(400),
                transforms.ToTensor(),  # 转为Tensor
                             ])
        else:
            self.trans_mask = transform_mask
        self.img = self.preprocess(self.img_base_path, self.img_set, False)
        self.mask = self.preprocess(self.mask_base_path, self.mask_set, True)

    def __len__(self):
        return len(self.img)

    def preprocess(self, base_path, img_set, is_mask):
        crop_size = self.crop_size
        result = []
        for name in img_set:
            path = join(base_path, name)
            img = open_image_np(path).astype(int)
            h = math.ceil(img.shape[0] / crop_size)
            w = math.ceil(img.shape[1] / crop_size)
            img0 = h * crop_size
            img1 = w * crop_size
            if is_mask:
                new = np.zeros([img0, img1], dtype=int)
                new[0: img.shape[0], 0: img.shape[1]] = img / 255
                new = new.astype('float')
                if self.mode == "test":
                    for row in range(0, img0, crop_size):
                        for col in range(0, img1, crop_size):
                            pil_img = np_to_pil(new[row: row + crop_size, col: col + crop_size])
                            trans_img = self.trans_mask(pil_img)
                            result.append(trans_img)
                else:
                    for row in range(0, img0 - (crop_size // 2), crop_size // 2):
                        for col in range(0, img1 - (crop_size // 2), crop_size // 2):
                            pil_img = np_to_pil(new[row: row + crop_size, col: col +  crop_size])
                            trans_img = self.trans_mask(pil_img)
                            result.append(trans_img)
            else:
                assert img.ndim == 3, f'image is not RGB'
                if img.ndim == 3:
                    new = np.zeros([img0, img1, 3], dtype=np.uint8)
                    new.fill(245)
                    new[0: img.shape[0], 0: img.shape[1], :] = img
                    if self.mode == "test":
                        for row in range(0, img0, crop_size):
                            for col in range(0, img1, crop_size):
                                pil_img = np_to_pil(new[row: row + crop_size, col: col + crop_size, :])
                                trans_img = self.trans_img(pil_img)
                                result.append(trans_img)
                    else:
                        for row in range(0, img0-(crop_size//2), crop_size // 2):
                            for col in range(0, img1-(crop_size // 2), crop_size // 2):
                                pil_img = np_to_pil(new[row: row + crop_size, col: col + crop_size, :])
                                trans_img = self.trans_img(pil_img)
                                result.append(trans_img)
                else:
                    pass

        return result

    def __getitem__(self, idx):  # img:(crop_size, crop_size, 3) mask:(crop_size, crop_size)
        return self.img[idx], self.mask[idx]


if __name__ == '__main__':
    args = get_args()
    start_time = time.time()
    train_img_path = "Data/train/thumb"
    train_mask_path = "Data/train/mask"
    val_img_path = "Data/val/thumb"
    val_mask_path = "Data/val/mask"
    train_set = MyDataset(train_img_path, train_mask_path, crop_size=args.crop_size)
    val_set = MyDataset(val_img_path, val_mask_path, crop_size=args.crop_size)
    # img, mask = train_set[0]
    # print(img.shape, mask.shape)
    print(train_set.__len__())
    print(val_set.__len__())
#     for img, mask in train_set:
#         print(img.shape, mask.shape)
#         # count = Counter
#         # print('count', count(mask.numpy().reshape(-1)))
#     # print(img, mask)
#     # count = Counter
#
#     # # print(img.shape, mask.shape)
#     # # print(type(img))
#     # train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
#     # for img, mask in train_loader:
#     #     print(img.shape, mask.shape)
#     #     print(img, mask)
    print('running time', time.time() - start_time)