import os
import random
import staintools

import numpy as np

from PIL import Image
from torchvision import transforms


def prepare_macenko_normalizer(target_path):
    print(f"读取并配置转换目标域")
    target_img = Image.open(target_path)
    if target_img.mode != 'RGB':
        target_img = target_img.convert('RGB')

    # target_img = staintools.LuminosityStandardizer.standardize(np.array(target_img))
    target_img = np.array(target_img)
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(target_img)
    return normalizer


# 修改 Macenko 染色归一化类以接受预配置的染色归一化器
class MacenkoStainNormalization(object):
    def __init__(self, prob, normalizer):
        self.prob = prob
        self.normalizer = normalizer

    def __call__(self, img):
        if random.random() < self.prob:
            # 确保图像是uint8 RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # img = staintools.LuminosityStandardizer.standardize(np.array(img))
            img = np.array(img)
            transformed = self.normalizer.transform(img)
            return Image.fromarray(transformed)
        return img


code_path = os.path.dirname(__file__)
targeimg_path = os.path.join(code_path, "Ref.png")
normalizer = prepare_macenko_normalizer(targeimg_path)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
base_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])


augmentation_list = [
    transforms.Compose([
                transforms.Resize(224),
                MacenkoStainNormalization(prob=1, normalizer=normalizer),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    # transforms.Compose([
    #             transforms.Resize(224),
    #             # MacenkoStainNormalization(prob=1, normalizer=normalizer),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean, std)
    #         ])
]
