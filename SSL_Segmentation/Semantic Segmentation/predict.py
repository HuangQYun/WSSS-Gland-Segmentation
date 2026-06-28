import os.path
import time
import numpy as np
from myutil import *
from Model.Mymodel import Model
import torch
from Model.unet_model import UNet
from torchvision import transforms
from Evaluate_metric import dice_coeff
import torch.nn.functional as F
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

from collections import Counter


if __name__ == '__main__':
    # args = get_args()
    # crop_size = args.crop_size
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # path_list = ["D:/CSCC", "D:/ESCC", "D:/LUSCC", "D:/newtestFD", "D:/NPC", "D:/HNSCC", ]
    # path_list = ["D:/Semantic Segmentation/Data/test/thumb"]
    # for path in path_list:
    #     for root, dirs, files in os.walk(path):
    #         for name in files:
    #             thumb_path = os.path.join(root, name)
    #             if ".png" in name:
    #                 img = open_image_np(thumb_path)
    #                 crop_size = args.crop_size
    #                 h = math.ceil(img.shape[0] / crop_size)
    #                 w = math.ceil(img.shape[1] / crop_size)
    #                 img0 = h * crop_size
    #                 img1 = w * crop_size
    #                 new = np.zeros([img0, img1, 3], dtype=np.uint8)
    #                 new.fill(245)
    #                 new[0: img.shape[0], 0: img.shape[1], :] = img
    #                 start_time = time.time()
    #                 best_model = Model(args.input_channal, args.classes)
    #                 model_file = "epoch_9"
    #                 load_checkpoint(os.path.join('Model', model_file + '.pth.tar'), best_model)
    #                 # load_checkpoint('best' + '.pth.tar', best_model)
    #                 fill_mask = np.zeros([img0, img1], dtype=np.uint8)
    #                 best_model.eval()
    #                 for row in range(h):
    #                     for col in range(w):
    #                         trans_img = transform(
    #                             new[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size, :])
    #                         trans_img = trans_img.unsqueeze(0).to(device)
    #                         output = best_model(trans_img)
    #                         predict = output.squeeze().detach().numpy()
    #                         block_mask = np.zeros([crop_size, crop_size], dtype=np.uint8)
    #                         block_mask[predict > 0.5] = 255
    #                         fill_mask[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size] = block_mask
    #                 pen_mask = fill_mask[0: img.shape[0], 0: img.shape[1]]
    #                 pil_mask = np_to_pil(pen_mask)
    #                 pred_path = os.path.join("Data/test", model_file + "_pred")
    #                 make_dir(pred_path)
    #                 pil_mask.save(os.path.join(pred_path, name + ".png"))
    # for path in path_list:
    #     for root, dirs, files in os.walk(path):
    #         for name in files:
    #             if '.svs' in name:
    #                 slide_path = os.path.join(root, name)
    #                 # sl = getinfo_from_slide(path)
    #                 # print(slide_path)
    #                 thumb = save_thumbnail_img(slide_path, "All_svs")
    #                 img = pil_to_np_rgb(thumb)
    #                 h = math.ceil(img.shape[0] / crop_size)
    #                 w = math.ceil(img.shape[1] / crop_size)
    #                 img0 = h * crop_size
    #                 img1 = w * crop_size
    #                 new = np.zeros([img0, img1, 3], dtype=np.uint8)
    #                 new.fill(245)
    #                 new[0: img.shape[0], 0: img.shape[1], :] = img
    #                 start_time = time.time()
    #                 best_model = UNet(args.input_channal, args.classes)
    #                 load_checkpoint(os.path.join('Model', 'best' + '.pth.tar'), best_model)
    #                 # load_checkpoint('best' + '.pth.tar', best_model)
    #                 fill_mask = np.zeros([img0, img1], dtype=np.uint8)
    #                 best_model.eval()
    #                 for row in range(h):
    #                     for col in range(w):
    #                         trans_img = transform(
    #                             new[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size, :])
    #                         trans_img = trans_img.unsqueeze(0).to(device)
    #                         output = best_model(trans_img)
    #                         predict = output.squeeze().detach().numpy()
    #                         block_mask = np.zeros([crop_size, crop_size], dtype=np.uint8)
    #                         block_mask[predict > 0.5] = 255
    #                         fill_mask[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size] = block_mask
    #                 pen_mask = fill_mask[0: img.shape[0], 0: img.shape[1]]
    #                 pil_mask = np_to_pil(pen_mask)
    #                 pil_mask.save(os.path.join("All_svs", os.path.basename(slide_path).split('.')[0] + 'pen_mask.png'))
    #                 print('end time:', time.time() - start_time)

    # -----------------------------------------------------单个thumb----------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    img = open_image_np("Data/val/thumb/2019-01703N.png")
    # img = open_image_np("Data/val/thumb/2019-01037E.png")
    # img = open_image_np("Data/val/thumb/2019-64234A.png")
    # img = open_image_np("Data/train/thumb/2016-54291A.png")
    crop_size = args.crop_size
    h = math.ceil(img.shape[0] / crop_size)
    w = math.ceil(img.shape[1] / crop_size)
    img0 = h * crop_size
    img1 = w * crop_size
    new = np.zeros([img0, img1, 3], dtype=np.uint8)
    new.fill(245)
    new[0: img.shape[0], 0: img.shape[1], :] = img
    start_time = time.time()
    best_model = Model(args.input_channal, args.classes)
    load_checkpoint(os.path.join('Model', '9_4epoch_27' + '.pth.tar'), best_model)
    # load_checkpoint('best' + '.pth.tar', best_model)
    fill_mask = np.zeros([img0, img1], dtype=np.uint8)
    best_model.eval()
    for row in range(h):
        for col in range(w):
            trans_img = transform(
                new[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size, :])
            trans_img = trans_img.unsqueeze(0).to(device)
            output = best_model(trans_img)
            predict = output.squeeze().detach().numpy()
            block_mask = np.zeros([crop_size, crop_size], dtype=np.uint8)
            count = Counter
            # print(count(predict.reshape(-1)))
            block_mask[predict > 0.5] = 255
            fill_mask[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size] = block_mask
    pen_mask = fill_mask[0: img.shape[0], 0: img.shape[1]]
    # count = Counter
    # print(count(fill_mask.reshape(-1)))
    pil_mask = np_to_pil(pen_mask)
    pil_mask.show()
    print('end time:', time.time() - start_time)

    # ----------------------------------------------------------单个block--------------------------------------------
    # # mask = open_image_np("Data/train/mask/4S1757874-025_mask.png")
    # # true_mask = mask[100: 356, 100: 356]
    # test_img = img[100: 100+512, 100: 100+512, :]
    # pil_img = np_to_pil(test_img)
    # pil_img.show()

    # best_model.eval()
    #
    # test_img = transform(test_img)
    # test_img = test_img.unsqueeze(0).to(device)
    # # print(true_mask.shape)
    # output = best_model(test_img)
    # # print('output:', output)
    # print('output:', output.shape)
    # predict = output.squeeze().detach().numpy()
    # final_mask = np.zeros([512, 512], dtype=np.uint8)
    # final_mask[predict > 0.5] = 255
    # print('predict time:', time.time() - start_time)
    # pil_mask = np_to_pil(final_mask)
    # pil_mask.show()



