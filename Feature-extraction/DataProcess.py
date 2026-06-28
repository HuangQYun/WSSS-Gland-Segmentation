import os
import shutil
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import h5py
from pathlib import Path

def SplitSingoTilesFeatures(features_file_dir, save_features_file_dir):
    """
    前处理部分
    函数功能：将所有tiles的patch特征转换为单张tiles的patch
    tiles_dir：tiles坐在文件夹
    """
    def GetCoords():
        coords = []
        for i in range(1, 15):
            for j in range(1, 15):
                coords.append([j, i])
        return coords

    coords = GetCoords()
    for features_file in os.listdir(features_file_dir):
        file_data = h5py.File(f"{features_file_dir}/{features_file}", "r")
        file_coords_data = file_data["coords"][:]
        len_index = len(file_coords_data)
        save_dir = f"{save_features_file_dir}/{features_file[:-3]}"
        os.makedirs(save_dir, exist_ok=True)
        for index in range(len_index):
            save_file_feats_data = file_data["feats"][index]
            save_file_coords_data = file_data["coords"][index]
            tiles_features_dir = f"{save_dir}/{save_file_coords_data[1]}_{save_file_coords_data[0]}.h5"
            with h5py.File(tiles_features_dir, 'w') as f:
                f['coords'] = np.array(coords)
                f['feats'] = save_file_feats_data
                f['augmented'] = np.array([False] * len(coords))
                f.close()


def GetFinblockFromMsimap(msimap_dir, fin_block_dir):
    """
    函数功能：生成final_block
    """
    img_name_list = []

    def get_rc_from_img(img_name):
        res = img_name[:-4].split("_")[0]
        col = img_name[:-4].split("_")[1]
        point = (int(res), int(col))
        return point

    def get_all_save_block(msimap_dir, img_name_list):
        MSImap_data = Image.open(msimap_dir)
        MSImap_size = MSImap_data.size
        temp_img = np.zeros((MSImap_size[1], MSImap_size[0]), dtype=np.uint8)
        for i in img_name_list:
            point = get_rc_from_img(i)
            temp_img[point[0], point[1]] = 255
        return temp_img

    tiles_path = os.path.dirname(os.path.dirname(msimap_dir))
    for file in os.listdir(tiles_path):
        if file[-4:] == ".png":
            img_name_list.append(file)

    FinBlockData = get_all_save_block(msimap_dir, img_name_list)
    cv2.imwrite(fin_block_dir, FinBlockData)


def ExtendPic(Image_path, write_pic_dir):
    """
    函数功能：修正final_block错误，同时转换背景和组织对应的颜色。
    """

    def fix_pic(Image_path):
        # 读取图像
        image = cv2.imread(Image_path)
        # 获取图像的高度和宽度
        height, width, _ = image.shape
        # 创建副本以修改图像
        modified_image = np.copy(image)
        # 剪切第一行并粘贴到最后一行
        modified_image[-1, :] = image[0, :]
        # 删除原始的第一行（现在成为第二行），将后面的行向上移动
        modified_image[0:-1, :] = image[1:, :]
        # 剪切第一列并粘贴到最后一列
        modified_image[:, -1] = modified_image[:, 0]
        # 删除原始的第一列（现在成为第二列），将后面的列向左移动
        modified_image[:, 0:-1] = modified_image[:, 1:]
        return modified_image

    def expand_pixels_fast(image_array, block_size=14):
        return np.repeat(np.repeat(image_array, block_size, axis=0), block_size, axis=1)

    # 读取图像并转换文件类型
    img_data = fix_pic(Image_path)
    image_array = np.array(img_data)
    # 执行高效的像素扩展
    expanded_image_array_fast = expand_pixels_fast(image_array)

    # 转换回图像
    expanded_image_fast = Image.fromarray(expanded_image_array_fast)
    expanded_image_fast = np.array(expanded_image_fast)
    F_data = cv2.bitwise_not(expanded_image_fast)
    cv2.imwrite(write_pic_dir, F_data)


def get_probability(predict_heatmap_dir, save_predict_heatmap_dir):
    os.makedirs(save_predict_heatmap_dir, exist_ok=True)
    for file in os.listdir(predict_heatmap_dir):
        slect_str = "PROBABILITY_heatmap.png"
        if slect_str in file:
            select_pic_dir = f"{predict_heatmap_dir}/{file}"
            save_pic_dir = f"{save_predict_heatmap_dir}/{file}"
            shutil.copy(select_pic_dir, save_pic_dir)


def montage_heatmap(PredictFile, Heatmap_dir, extend_bid_finblock, final_image_path):
    num = 0
    # 定义一个函数将图像文件名的行列对调，并将其从1开始的坐标转为0开始
    def swap_and_adjust_coords(filename):
        row, col = filename.split('_')
        return int(col) - 1, int(row) - 1  # 调整为从0开始

    # 应用该函数到PATIENT列，进行行列对调并调整为0开始的坐标
    csv_data = pd.read_csv(PredictFile)
    csv_data['adjusted_coords'] = csv_data['PATIENT'].apply(swap_and_adjust_coords)
    expanded_image = Image.open(extend_bid_finblock)
    # 获取扩展后的图像大小
    expanded_image_size = expanded_image.size
    # 创建一个新的大图，尺寸与扩展后的图像相同
    big_image = Image.new('RGB', expanded_image_size)
    # 将扩展后的图像作为底图
    big_image.paste(expanded_image)
    block_size = 14

    # 循环处理每个图像块
    for index, row in csv_data.iterrows():
        # 获取图像块的调整后坐标
        coord_row, coord_col = row['adjusted_coords']

        # 计算在扩展图像中的位置
        position = ((coord_row) * block_size, (coord_col) * block_size)

        # 生成要读取的图块的路径，例如'all_tiles/8_43.png'
        tile_filename = f"{coord_col + 1}_{coord_row + 1}_PROBABILITY_heatmap.png"  # 从0转回1开始的坐标
        tile_path = os.path.join(Heatmap_dir, tile_filename)

        # 如果图像块存在，则将它粘贴到大图上
        if os.path.exists(tile_path):
            num = num + 1
            tile_image = Image.open(tile_path)
            big_image.paste(tile_image, position)
        else:
            print(tile_path)
    big_image.save(final_image_path)


def get_resize_lr(lr_dir, montage_heatmap_dir, SaveResizeLRPath):

    lr_data = cv2.imread(lr_dir)
    montage_heatmap_data = cv2.imread(montage_heatmap_dir)
    lr_resize_data = cv2.resize(lr_data,
                                (montage_heatmap_data.shape[1], montage_heatmap_data.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(SaveResizeLRPath, lr_resize_data)


def addWeightedSmallImgToLargeImg(largeImg, alpha, smallImg, beta, gamma=0.0, regionTopLeftPos=(0,0)):
    srcW, srcH = largeImg.shape[1::-1]
    refW, refH = smallImg.shape[1::-1]
    x, y = regionTopLeftPos
    if (refW>srcW) or (refH>srcH):
        #raise ValueError("img2's size must less than or equal to img1")
        raise ValueError(f"img2's size {smallImg.shape[1::-1]} must less than or equal to img1's size {largeImg.shape[1::-1]}")
    else:
        if (x+refW)>srcW:
            x = srcW-refW
        if (y+refH)>srcH:
            y = srcH-refH
        destImg = np.array(largeImg)
        tmpSrcImg = destImg[y:y+refH,x:x+refW]
        tmpImg = cv2.addWeighted(tmpSrcImg, alpha, smallImg, beta, gamma)
        destImg[y:y + refH, x:x + refW] = tmpImg
        return destImg

def ProcessStart(AllTiles, AllTilesPatchFeatures, SaveReslutPath):
    for patient in os.listdir(AllTiles):
        msimap_dir = f"{AllTiles}/{patient}/LRimages/MSImap.jpg"
        fin_block_dir = f"{AllTiles}/{patient}/LRimages/finblock.png"

        # step1 生成finblock图像
        GetFinblockFromMsimap(msimap_dir, fin_block_dir)

        # step2 修正finblock大图，并放大
        finblock_big_dir = f"{AllTiles}/{patient}/LRimages/finblock_big.png"
        ExtendPic(fin_block_dir, finblock_big_dir)

        # step3 将patch的热力图拼接到step2生成的大图中。获取整张slide的热力图
        PredictFile = Path(AllTilesPatchFeatures/patient/"patient-preds.csv")
        ProbabilityHeatmapPath = Path(AllTilesPatchFeatures/patient/"HEATMAP")
        SaveResultPath = f"{AllTiles}/{patient}/LRimages/Result.png"
        montage_heatmap(PredictFile, ProbabilityHeatmapPath, finblock_big_dir, SaveResultPath)

        # step4 将LR缩略图缩放到与热力图相同大小用于后续叠加
        lr_dir = f"{AllTiles}/{patient}/LRimages/LR.jpg"
        SaveResizeLRPath = f"{lr_dir[:-4]}_resize.png"
        get_resize_lr(lr_dir, SaveResultPath, SaveResizeLRPath)

        # step5 将缩略图与热力图叠加
        OverlayResult = f"{SaveReslutPath}/{patient}_Result.png"
        image1 = cv2.imread(SaveResizeLRPath)
        image2 = cv2.imread(SaveResultPath)
        img = addWeightedSmallImgToLargeImg(image1, 0.6, image2, 0.4)
        cv2.imwrite(OverlayResult, img)

