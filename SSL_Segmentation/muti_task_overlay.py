import os
import torch

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from model._factory import create_model
from model.SC_head import MulTaskHead
from utils import overlay_boundary, overlay_mask, plot_comparison


prov_gigapath_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def inference_image(torch_feats, net):
    b_feats = torch_feats.unsqueeze(0)

    seg_output, cls_output = net(b_feats)
    prob = torch.softmax(seg_output, dim=1)
    predicted = torch.where(prob[:, 1, :, :] > 0.5, True, False)
    predict_mask = predicted.squeeze().cpu().numpy()
    predict_mask = predict_mask.astype(np.uint8)
    return predict_mask


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ====================================== load gigapath =============================================
    json_path = "D:/Deeplearning_weights/prov-gigapath/config.json"  # gigapath 配置文件
    backbone_pretrain_path = "D:/Deeplearning_weights/prov-gigapath/pytorch_model.bin"  # gigapath 权重
    backbone_pretrain_weights = torch.load(backbone_pretrain_path)
    backbone = create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,
                            json_path=json_path).to(device)
    backbone.load_state_dict(backbone_pretrain_weights)
    backbone.eval()

    # ====================================== load head =============================================
    seg_head = MulTaskHead().to(device)
    weight_path = "weight/best.pth"

    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    seg_head.load_state_dict(weight)
    seg_head.eval()

    Test_path = 'D:/SHARER/Data/肠黏膜数据集/Test1'  # 图像输入文件夹
    out_path = 'D:/SHARER/Data/肠黏膜数据集/Test1_Seg_Results/'  # 输出文件夹

    repeat_channel = (lambda x: np.repeat(x, 3, axis=-1))

    os.makedirs(out_path, exist_ok=True)
    suffix_img = ['png', 'jpg', 'tif', 'bmp']
    for img_name in tqdm(os.listdir(Test_path)):

        if os.path.isdir(os.path.join(Test_path, img_name)):
            continue
        patient, suffix = img_name.split('.')
        if suffix not in suffix_img:
            continue
        save_name = patient + '-overlay'
        if os.path.exists(os.path.join(out_path, save_name + '.png')):
            continue
        original_image = Image.open(os.path.join(Test_path, img_name))
        original_np_img = np.array(original_image)

        img_tensor = prov_gigapath_transform(original_image)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        feats = backbone.forward_features(img_tensor)

        mask = inference_image(feats, seg_head)
        mask = mask * 255
        mask = repeat_channel(mask[:, :, np.newaxis])
        resize_mask = Image.fromarray(mask).resize(original_image.size)
        final_mask = np.array(resize_mask)
        pred_overlay_img = overlay_boundary(overlay_mask(original_np_img, final_mask, alpha=0.1),
                                            final_mask)
        imgs = [original_np_img, final_mask, pred_overlay_img]
        captions = ['Gland Image', 'Mask', 'Overlay']
        plot_comparison(imgs, captions, n_col=len(imgs), figsize=(12, 12), cmap=None,
                        plot=False, save_path=out_path, save_name=save_name)


