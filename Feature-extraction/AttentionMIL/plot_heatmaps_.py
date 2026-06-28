import os.path
from enum import Enum, auto
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd
from fastai.vision.learner import load_learner
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from FeaturesbasedMIL.mil.data import get_target_enc
from matplotlib.patches import Patch
from scipy import interpolate
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import glob
import cv2
import re


colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])
class MapType(Enum):
    ATTENTION = auto()
    PROBABILITY = auto()
    CONTRIBUTION = auto()


def _get_slide_features(h5_feature_dir, ws_path):

    assert h5_feature_dir.is_dir(), \
        f'{h5_feature_dir} is not a directory. Please provide path to feature directory!'

    whole_slides = []
    h5_feature_paths = []
    lr_list = []
    if ws_path.is_file():
        whole_slides.append(ws_path)
        h5_feature_path = h5_feature_dir/ws_path.with_suffix('.h5').name
        h5_feature_paths.append(h5_feature_path)
    elif ws_path.is_dir():
        for p in os.listdir(ws_path):
            # # 原始程序修改处
            # lr_list.append(f"{ws_path}/{p}/LRimages/LR.jpg")
            lr_list.append(f"{ws_path}/{p}")

        for LR_img_path in lr_list:
            # # 原始程序修改处
            # PID = get_patient_name(os.path.dirname(LR_img_path))
            PID = os.path.splitext(os.path.basename(LR_img_path))[0]
            h5_feature_path = h5_feature_dir/(str(PID) + ".h5")
            if h5_feature_path.is_file():
                whole_slides.append(LR_img_path)
                h5_feature_paths.append(h5_feature_path)
            else:
                print(
                    f'Could not find file {h5_feature_path}.\
                         Check if features where extracted for this whole slide image!')
    else:
        raise ValueError(
            f'Given ws_path is neither a file nor a directory. Path given {ws_path=!r}.')

    return list(zip(whole_slides, h5_feature_paths))


def get_true_coords(coords):
    for i in range(len(coords)):
        coords[i][0] = coords[i][0] - 1
        coords[i][1] = coords[i][1] - 1
    return coords


def get_dict_maptype_to_coords_scores(h5_feature_path: Path, model: nn.Module,map_types
                                      #map_types: list[MapType],
                                      ) -> dict:
    # map_types = [MapType.CONTRIBUTION]
    dict_maptype_to_coords_scores = {}

    feats, coords, sizes = [], [], []
    with h5py.File(h5_feature_path, 'r') as f:
        feats.append(torch.from_numpy(f['feats'][:]).float())
        sizes.append(len(f['feats']))
        # coords.append(torch.from_numpy(f['coords'][:]))
        tiles_cooeds = get_true_coords(torch.from_numpy(f['coords'][:]))
        coords.append(tiles_cooeds)
    feats, coords = torch.cat(feats), torch.cat(coords)

    encoder = model.encoder.eval()
    attention = model.attention.eval()
    head = model.head.eval()

    # calculate attention, scores etc.
    encs = encoder(feats)
    patient_atts = torch.softmax(attention(encs), dim=0).detach()
    patient_scores = torch.softmax(head(encs), dim=1).detach()
    normed_patient_atts=(patient_atts-patient_atts.min())/(patient_atts.max()-patient_atts.min())
    patient_weighted_scores=normed_patient_atts*patient_scores

    assert patient_scores.shape[-1] <= colors.shape[0], f'not enough colours.\n'\
        'Can only plot score for max {colors.shape[0]}'\
        'classes at a time!\n Number of classes asked for:'\
        f'{len()} not supported.'
    for map_type in map_types:
        if (map_type==MapType.ATTENTION):
            scores = patient_atts.numpy()
            scores -= scores.min()
            scores /= (scores.max()-scores.min())
        elif(map_type==MapType.PROBABILITY):
            scores = patient_scores.numpy()
        elif(map_type == MapType.CONTRIBUTION):
            scores = patient_weighted_scores.numpy()
        else:
            raise ValueError(f'heat map type {map_type} not supported!')

        dict_maptype_to_coords_scores[map_type] = coords.numpy(), scores

    return dict_maptype_to_coords_scores


def _visualize_activation_map(activations: np.ndarray, colors: np.ndarray, alpha: float = 1.,
    clipping: bool=True) -> np.ndarray:
    """Transforms an activation map into an RGBA numpy array.
    Args:
        activations: An (h, w, D) array of activations.
        colors: A (D, 3) array mapping each of the target classes to a color.
    Returns:
        An interpolated activation map. Regions which the algorithm assumes to be background
        will be transparent.
    """
    assert colors.shape[1] == 3, "expected color map to have three color channels"
    assert colors.shape[0] == activations.shape[2], "one color map entry per class required"
    # activations should be less or equal to 1
    assert activations[2].max() <= 1, f"Activations should be less than one, otherwise maps get clipped! \n \
        Max value provided {activations[2].max()}."
    # transform activation map into RGB map
    rgbmap = activations.dot(colors)
    # TODO this is a cheap fix only!
    if clipping:
        max_cvalue=np.amax(rgbmap)
        if max_cvalue>255.0:
            #Rescale
            rgbmap=rgbmap/max_cvalue*255.0
            print(f"Rescaled rgbmap as max pixel value is {max_cvalue}")
            print(f"This could potentially be a problem!")

    # create RGBA map with non-zero activations being the foreground
    mask = activations.any(axis=2)

    # mask * alpha gives alpha at non zero values of activation
    # below gives value for the alpha channel
    im_data = (np.concatenate([rgbmap, np.expand_dims(
        mask * alpha, -1)], axis=2) * 255.5).astype(np.uint8)

    return im_data


def _get_stride(coordinates: np.ndarray) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)

    return stride


def _MIL_heatmap_for_slide(coords: np.ndarray, scores: np.ndarray, LR_img_path: Path, h5_feature_dir: Path,
                           colours: np.ndarray = None) -> np.ndarray:
    """
    Args:
        h5_feature_path: path to .h5 file with features to analyse
        model: model to analyse slide with
        categories: TODO
        map_type: one from ['attention','probability', contribution]

    Returns:
        Tuple of covered_area, legend, heatmap
        covered_area: extent in x dimension and extent in y dimension of whole slide image
        legend: details of legend for plot
        heatmap: the actual heatmap, z coordinate is activation, x and y are integers from 0 to n_x
            and 0 to n_y, respectively. To get pixel dimensions multiply x and y by stride
    """

    # get stride
    stride = _get_stride(coords)
    scaled_map_coords = coords // stride
    if colours is not None:
        pass
    else:
        colours = colors

    # make a mask, 1 where coordinates have attention 0 otherwise
    # ndarray of zeros of dimension max_x * max_y
    mask = np.zeros(scaled_map_coords.max(0) + 1)
    # add in ones where we have values
    for coord in scaled_map_coords:
        mask[coord[0], coord[1]] = 1
    # # 修改之前
    # grid_x, grid_y = np.mgrid[0:scaled_map_coords[:, 0].max()+1,
    #                           0:scaled_map_coords[:, 1].max()+1]

    grid_x, grid_y = np.mgrid[0:scaled_map_coords[:, 1].max()+1,
                              0:scaled_map_coords[:, 0].max()+1]

    # interpolate heatmap over grid
    if scores.ndim < 2:
        scores = np.expand_dims(scores, 1)
    activations = interpolate.griddata(
        scaled_map_coords, scores, (grid_x, grid_y))
    activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

    heatmap = _visualize_activation_map(
        activations.transpose(1, 0, 2), colours[:activations.shape[-1]])
    # print(heatmap.shape)
    # patient_name = os.path.basename(LR_img_path)[:-7]
    # heatmap_image = Image.fromarray(heatmap.astype(np.uint8), "RGBA")
    # heatmap_save_path = str(h5_feature_dir) + "/" + patient_name + "_22heatmap.png"
    # heatmap_image.save(heatmap_save_path)
    heatmap = heatmap[:, :, [2, 1, 0, 3]]

    return heatmap

def plot_heatmaps_(out_dir: Path, model_path: Path, tiles_path: Path, h5_feature_dir: Path,map_types,
                  #map_types: list[MapType],
                  superimpose: bool = True, alpha: float = 0.5):
    """Generates heatmaps for whole slide images.

    Outputs heatmaps to project directory, in subfolders for each map_type.

    Args:
        out_dir: path to where outputs are stored
        train_dir: path to directory where training was done, i.e. where export.pkl file is located
        ws_path: path to whole slide image, either full path -> single image is analysed or directory
            -> all whole slide images in directory are analysed
        h5_feature_dir: directory containing features used in training, must match whole slide images
        map_types: list containing attention, probability and/or contribution to give corresponding heatmaps
        superimpose: have heatmap on top of thumbnail or both side-by-side
        alpha: transparacy of heatmap
    """
    for file in os.listdir(h5_feature_dir):
        deal_features_dir = Path(h5_feature_dir/file)
        deal_tiles_dir = Path(tiles_path / file)
        slide_features = _get_slide_features(deal_features_dir, deal_tiles_dir)
        learn = load_learner(model_path)
        for LR_img_path, h5_feature_path in slide_features:
            try:
                dict_maptype_to_coords_scores = get_dict_maptype_to_coords_scores(h5_feature_path,
                                                                                  model=learn.model, map_types=map_types)
                for map_type in map_types:
                    coords, scores = dict_maptype_to_coords_scores[map_type]
                    heatmap = _MIL_heatmap_for_slide(coords=coords, scores=scores,
                                                     LR_img_path=LR_img_path, h5_feature_dir=deal_features_dir)
                    plot_heatmaps(heatmap, LR_img_path, deal_features_dir, map_type)
            except:
                print(f"处理文件失败：{LR_img_path}")


def plot_heatmaps(heatmap, LR_path, h5_feature_dir, map_type):
    '''
    函数功能：将生成的热力图与保存tiles时生成的缩略图叠加
    heatmap：四通道热力图
    LR_path：保存tiles生成的缩略图路径
    h5_feature_dir：h5文件的储存位置，也是保存结果图的路径
    map_type：热力图类型列表，只是为了获取类型热力图的后缀名用以区别保存
    '''
    model_type = str(map_type).split(".")[1]
    # # 修改之前程序
    # PID = get_patient_name(os.path.dirname(LR_path))
    # msi_map_path = get_MSImap_path(LR_path)
    PID = os.path.splitext(os.path.basename(LR_path))[0]
    save_result_path = str(h5_feature_dir) + "/HEATMAP"

    if os.path.exists(save_result_path):
        pass
        # print(f"The folder where the results are saved already exists, and the folder creation step will be skipped!!!")
    else:
        os.makedirs(save_result_path)
    heatmap_save_path = save_result_path + "/" + PID + "_" + model_type + "_heatmap.png"
    # four_channel_thumb_path = save_result_path + "/" + os.path.basename(LR_path)[:-7] + "_four_channel_thumb.png"
    overlay_save_img = save_result_path + "/" + PID + "_" + model_type + "_overlay_thumb.png"

    # # 修改之前
    # ori_heatmap = get_ori_heatmap(msi_map_path, heatmap)
    ori_heatmap = get_ori_heatmap("", heatmap)
    change_channel_heatmap = change_heatmap_channel(ori_heatmap)
    thumb_img = cv2.imdecode(np.fromfile(f"{LR_path}", dtype=np.uint8), cv2.IMREAD_COLOR)
    four_channel_thumb = change_thumb_channel(thumb_img)
    overlay_img = image_overlay(change_channel_heatmap, four_channel_thumb)
    cv2.imencode('.png', overlay_img)[1].tofile(overlay_save_img)
    cv2.imencode('.png', heatmap)[1].tofile(heatmap_save_path)
    if map_type == MapType.CONTRIBUTION:
        msi_img = str(h5_feature_dir) + "/" + PID + "_MSI.jpg"
        thumb_img_name = str(h5_feature_dir) + "/" + PID + "_LR.jpg"
        cv2.imencode('.jpg', overlay_img)[1].tofile(msi_img)
        cv2.imencode('.jpg', thumb_img)[1].tofile(thumb_img_name)

def get_ori_heatmap(msi_map_path, heatmap):
    '''
    该函数将生成的热力图还原到与msimap相同的大小，方便与缩略图叠加
    msi_map_path：mismap的路径，在读取时会以二值图的形式读取
    heatmap:四通道热力图
    返回值：与msimap长宽相同的四通道热力图，填充的方式(下方填充，右方填充)
    '''
    # 修改之前
    # ori_msi_map = cv2.imdecode(np.fromfile(msi_map_path, dtype=np.uint8), -1)
    #
    # top_size, bottom_size, left_size, right_size = (0, ori_msi_map.shape[0] - heatmap.shape[0],
    #                                                 0, ori_msi_map.shape[1] - heatmap.shape[1])

    top_size, bottom_size, left_size, right_size = (0, 0, 0, 0)
    ori_heatmap = cv2.copyMakeBorder(heatmap, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT,
                                      value=0)
    return ori_heatmap

def change_heatmap_channel(ori_heatmap):
    '''
    函数功能：将热力图中的黑色像素与热力值部透明度分开，防止黑色像素叠加时导致叠加图像整体偏暗的情况
    ori_heatmap：与msomap相同长宽的四通道热力图
    '''
    black_pixels = np.all(ori_heatmap[:, :, :3] == [0, 0, 0], axis=2)
    ori_heatmap[black_pixels, 3] = 0  # 设置黑色像素为完全透明
    four_channel_heatmap = ori_heatmap
    return four_channel_heatmap

def change_thumb_channel(thumb_img):
    '''
    将保存的缩略图由三通道转为四通道
    '''
    alpha_channel = np.ones(thumb_img.shape[:2], dtype=thumb_img.dtype) * 255
    # 合并三通道图像与alpha通道
    four_channel_thumb = cv2.merge([thumb_img[:,:,0], thumb_img[:,:,1], thumb_img[:,:,2], alpha_channel])
    return four_channel_thumb

def image_overlay(change_channel_heatmap, four_channel_thumb):
    '''
    四通道的热力图与四通道的缩略图进行叠加
    '''
    resized_small = cv2.resize(change_channel_heatmap, (four_channel_thumb.shape[1], four_channel_thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    for i in range(3):  # 对于B, G, R三个通道
        four_channel_thumb[:, :, i] = four_channel_thumb[:, :, i] * (1 - resized_small[:, :, 3] / 255.0) + resized_small[:, :, i] * (resized_small[:, :, 3] / 255.0)
    # 保存合成后的图像
    return four_channel_thumb

def get_MSImap_path(LR_path):
    parent_path = os.path.dirname(LR_path)
    file_list = os.listdir(parent_path)
    regex = re.compile("MSImap.jpg")
    results = [file_name for file_name in file_list if regex.search(file_name)]
    out_path = parent_path + "/" + results[0]
    return out_path