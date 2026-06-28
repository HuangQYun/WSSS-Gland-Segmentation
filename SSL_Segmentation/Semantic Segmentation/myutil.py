import numpy as np
from PIL import Image
import os
from collections import Counter
import math
from read_chinese import getinfo_from_slide
import openslide as ops
from skimage.morphology import remove_small_objects
import scipy.ndimage.morphology as sc_morph
import skimage.morphology as sk_morphology
import time
import logging
import shutil
import torch
import argparse
from os.path import join
logger_fmt = '%(asctime)s - %(funcName)s - %(message)s'
import openslide


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=20, help='Batch size')
    parser.add_argument('--crop_size', '-cs', dest='crop_size', metavar='B', type=int, default=512, help='Crop size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default='Model', help='Load model from a .pth file')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--input_channal', '-ic', type=int, default=3)
    return parser.parse_args()


logger = logging.getLogger('logger')


def Log(name, log_name):
    log_path = './log'
    make_dir(log_path)
    train_logger = logging.getLogger(name)
    train_logger.setLevel(logging.INFO)
    log_handle = logging.FileHandler(join(log_path, log_name + ".log"), mode='a')
    formatter = logging.Formatter(logger_fmt)
    log_handle.setFormatter(formatter)
    train_logger.addHandler(log_handle)

    return train_logger


def save_checkpoint(state, is_best, epoch, load_path, sava_epoch, ins_name=-1):
    '''
        Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
        load_path + 'best.pth.tar'
        Args:
            state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
            is_best: (bool) True if it is the best model seen till now
            load_path: (string) folder where parameters are to be saved
            ins_name: (int) instance index
    '''
    if sava_epoch:
        if ins_name == -1:
            filepath = os.path.join(load_path, f'epoch_{epoch}.pth.tar')
        else:
            filepath = os.path.join(load_path, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
        if not os.path.exists(load_path):
            logger.info(f'Checkpoint Directory does not exist! Making directory {load_path}')
            os.mkdir(load_path)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(load_path, 'best.pth.tar'))
            logger.info('Best checkpoint copied to best.pth.tar')
    else:
        best_path = os.path.join(load_path, 'best.pth.tar')
        make_dir(best_path)
        if is_best:
            torch.save(state, best_path)


def load_checkpoint(path, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])


def turn_um(width, height, path):
    slide = getinfo_from_slide(path)
    mppx_available = False
    mppy_available = False
    try:
        if ops.PROPERTY_NAME_MPP_X in slide.properties:
            MPPX = float(slide.properties[ops.PROPERTY_NAME_MPP_X])
            mppx_available = True
        elif 'tiff.XResolution' in slide.properties:
            MPPX = 1 / float(slide.properties['tiff.XResolution']) * 10000
            mppx_available = True
    except KeyError:
        print("MPPX information is not available in the slide properties")  # 切片属性中没有MPPX信息
    try:
        if ops.PROPERTY_NAME_MPP_Y in slide.properties:
            MPPY = float(slide.properties[ops.PROPERTY_NAME_MPP_Y])
            mppy_available = True
        elif 'tiff.YResolution' in slide.properties:
            MPPY = 1 / float(slide.properties['tiff.YResolution']) * 10000
            mppy_available = True
    except KeyError:
        print("MPPY information is not available in the slide properties")  # 切片属性中没有MPPY信息
    if mppx_available and mppy_available:  # 增加判断，mppx mppy只要有一个没有，就仍以pixel为tile的单位
        width = int(width / MPPX)
        height = int(height / MPPY)

    return width, height


def getBestThumb(osh, img_base_size):
    dims = tuple(np.rint(np.asarray(img_base_size)).astype(int))  # dims是一个长宽两个值，代表缩略图的大小
    max_dim = dims[0] if dims[0] > dims[1] else dims[1]
    return osh.get_thumbnail((max_dim, max_dim)) #return后面不能加别的东西，否则不会保存缩略图


def get_rgb(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B


def save_thumbnail_img(slide_path, save_path):
    make_dir(save_path)
    thumb_path = os.path.join(save_path, os.path.basename(slide_path).split('.')[0] + '.png')
    if os.path.exists(thumb_path):
        img = open_image(thumb_path)
    else:
        slide = openslide.OpenSlide(slide_path)
        shape = slide.dimensions  # 获取切片原图的尺寸，即宽度和高度 (以像素为单位)。
        goal_thumb_area = 2048 * 2048  # 预设的缩略图面积，目标：使生成的缩略图的面积不超过如：4096 * 4096像素=16777216，最后的缩略图对宽和高没有要求，只对面积有要求
        y_x_ratio = shape[1] / shape[0]  # 1为列数-宽，0为行数-高，也就是切片原图宽/高（宽高比）
        thumb_x = math.sqrt(goal_thumb_area / y_x_ratio)  # 将缩略图面积与宽高比相除，计算出在缩略图宽高比固定的情况下，应该设置的缩略图宽度。sqrt求平方根
        thumb_y = thumb_x * y_x_ratio  # y/x=宽/高   thumb_y=宽  虽然这里定义了y为宽，x为高，但是谁大还不一定，只是确定位置，x是竖线长度，y是横线长度
        img_base_size = (thumb_x, thumb_y)
        img = getBestThumb(slide, img_base_size)
        img.save(thumb_path)
        slide.close()

    return img


def judge_Annotations(svs_path):
    basename = os.path.basename(svs_path) # svs_path 点击的文件路径
    basename = os.path.splitext(basename)[0] # 去掉后缀之后的名字
    up_file = os.path.dirname(svs_path) # 打开的输入文件夹的路径
    if os.path.exists(up_file + "/" + basename + "_kfb" + "/Annotations/" + "1.json") or os.path.exists(up_file + "/" + "CRC" + "/" + basename + ".csv"):
        return True
    return False


def count(path):
    pass
    # count = Counter
    # print(count(b))
    # print(count(b.tolist()))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).

    Args:
    filename: Name of the image file.

    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image


def open_image_np(filename):
    """
    Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

    Args:
    filename: Name of the image file.

    returns:
    A NumPy representing an RGB image.
    """
    pil_img = open_image(filename)
    if pil_img.mode == "RGB":
        np_img = pil_to_np_rgb(pil_img)
    elif pil_img.mode == "L":
        np_img = np.array(pil_img)
    return np_img


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
    pil_img: The PIL Image.
    Returns:
    The PIL image converted to a NumPy array.
    """
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.

    Shape (h, w, c) to (h, w).

    Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

    Returns:
    Grayscale image as NumPy array with shape (h, w).
    """
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    return grayscale


def filter_grays(rgb, tolerance=30, output_type="bool"):
    """
    Create a mask to filter_png out pixels where the red, green, and blue channel values are similar.

    Args:
      np_img: RGB image as a NumPy array.
      tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result
