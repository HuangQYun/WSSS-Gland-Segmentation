import h5py
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch import nn


# def get_attention_score(h5_feature_path: Path, model: nn.Module):
#     feats, coords, sizes = [], [], []
#     with h5py.File(h5_feature_path, 'r') as f:
#         feats.append(torch.from_numpy(f['feats'][:]).float())
#         sizes.append(len(f['feats']))
#         coords.append(torch.from_numpy(f['coords'][:]))
#     feats, coords = torch.cat(feats), torch.cat(coords)
#     #将坐标转换为图像名称
#     coords = coords.numpy()
#     coodrs_str = np.core.defchararray.add(coords[:, 1].astype(str), "_")
#     coodrs_str = np.core.defchararray.add(coodrs_str, coords[:, 0].astype(str))
#     image_name = np.core.defchararray.add(coodrs_str, ".png")
#
#     encoder = model.encoder.eval()
#     attention = model.attention.eval()
#     # calculate attention, scores etc.
#     encs = encoder(feats)
#     patient_atts = torch.softmax(attention(encs), dim=0).detach()
#     normed_patient_atts = (patient_atts - patient_atts.min()) / (patient_atts.max() - patient_atts.min())
#     attention_score = {name: att.item() for name, att in zip(image_name, normed_patient_atts.numpy())}
#     return attention_score


def get_attention_score(h5_feature_path: Path, model: nn.Module):
    feats, coords, sizes = [], [], []
    with h5py.File(h5_feature_path, 'r') as f:
        feats.append(torch.from_numpy(f['feats'][:]).float())
        sizes.append(len(f['feats']))
        coords.append(torch.from_numpy(f['coords'][:]))
    feats, coords = torch.cat(feats), torch.cat(coords)

    # 将坐标转换为图像名称
    coords = coords.numpy()
    coords_str = np.core.defchararray.add(coords[:, 1].astype(str), "_")
    coords_str = np.core.defchararray.add(coords_str, coords[:, 0].astype(str))
    image_name = np.core.defchararray.add(coords_str, ".png")

    # 模型各部分的计算
    encoder = model.encoder.eval()
    attention = model.attention.eval()
    head = model.head.eval()

    encs = encoder(feats)
    patient_atts = torch.softmax(attention(encs), dim=0).detach()
    patient_scores = torch.softmax(head(encs), dim=1).detach()
    normed_patient_atts = (patient_atts-patient_atts.min())/(patient_atts.max()-patient_atts.min())
    patient_weighted_scores = normed_patient_atts*patient_scores

    # 计算注意力得分
    attention_score = normed_patient_atts.squeeze()
    attention_score = attention_score.tolist()

    # 计算预测得分
    probability_list = list(patient_scores[:, 0].numpy())

    # 计算注意力加权得分
    patient_weight_one_score = patient_weighted_scores[:, 0].numpy()
    patient_weight_list = list(patient_weight_one_score)

    # 将结果存入DataFrame
    results_df = pd.DataFrame({
        'image_name': image_name,
        'attention_scores': attention_score,
        'probability': probability_list,  # 假设n_out是2
        'patient_weight': patient_weight_list
    })

    return results_df