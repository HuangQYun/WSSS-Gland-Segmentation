import os
import cv2
import timm
from tqdm import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from torchvision import transforms
from FeaturesbasedMIL._factory import create_model
from torch.utils.data import Dataset, DataLoader


def minmax_norm(x):
    """Min-max normalization"""
    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)


prov_gigapath_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.kind = [file for file in os.listdir(path)]
        self.base_transform = prov_gigapath_transform
        self.name, self.img_path = [], []
        self.patient_name = []
        for file_name in os.listdir(self.path):
            if file_name.endswith('.png') or file_name.endswith('jpg') or file_name.endswith('tif'):
                self.img_path.append(os.path.join(self.path, file_name))
                self.name.append(file_name)
                temp_name = os.path.splitext(file_name)[0]
                self.patient_name.append([int(temp_name.split("_")[1]), int(temp_name.split("_")[0])])

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx]).convert("RGB")
        base_tensor = self.base_transform(image)
        final_tensor = base_tensor
        return final_tensor, self.name[idx]


def generate_patch_norm_feature(dataloader, eval_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    patch_norm_feature = []

    for count, (image, name) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            image = image.to(device)
            batch_features = eval_model.forward_features(image)
            batch_features = batch_features.cpu().detach()
            patch_norm_feature.append(batch_features)
    patch_norm_feature = torch.concat(patch_norm_feature)
    return patch_norm_feature


def GetModel(model_name, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'uni':
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0,
            dynamic_img_size=True)
    else:
        model = create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, json_path="config.json")
    pretrain_weights = torch.load(weight_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()
    return model


def GetSlideFeature(input_path, out_path, model_name, weight_path, bs=64):
    os.makedirs(out_path, exist_ok=True)
    model = GetModel(model_name, weight_path)
    for bag in os.listdir(input_path):
        bag_out_path = f"{out_path}/{bag}.h5"
        bag_path = os.path.join(input_path, bag)
        image_set = ImageDataset(bag_path)
        i_n = image_set.patient_name
        if len(image_set) == 0:
            continue
        loader = DataLoader(image_set, batch_size=bs, shuffle=False, num_workers=0)
        batch_feature = generate_patch_norm_feature(loader, model)
        batch_feature = batch_feature[:, 1:, :]

        with h5py.File(bag_out_path, 'w') as f:
            t = (batch_feature.squeeze(0)).cpu().detach().numpy()
            f['coords'] = np.array(i_n)
            f['feats'] = t
            f['augmented'] = np.array([False]* (batch_feature.size()[1]))
            f.close()

# if __name__ == '__main__':
#     input_path = 'D:/20241114/tt'
#     out_path = 'D:/20241114/features/slide_all_tiles'
#     model_name = "gigapath"
#     weight_path = "D:/CanHelpCodes/ModelTraining/FeaturesbasedMIL0/weight/gigapath.bin"
#     GetSlideFeature(input_path,
#                     out_path,
#                     model_name,
#                     weight_path,
#                     bs=64)

