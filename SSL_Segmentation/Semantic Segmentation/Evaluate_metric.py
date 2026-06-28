import torch
from torch import Tensor
import numpy as np


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def calculate_TP(y, y_predict, T, N):
    TP = 0
    for i in range(len(y)):
        if y[i] == N and y_predict[i] == T:
            TP += 1

    return TP


def calculate_TN(y, y_predict, T, N):
    TN = 0
    for i in range(len(y)):
        if y[i] == N and y_predict[i] == N:
            TN += 1

    return TN


def calculate_FP(y, y_predict, T, N):
    FP = 0
    for i in range(len(y)):
        if y[i] == T and y_predict[i] == T:
            FP += 1

    return FP


def calculate_FN(y, y_predict, T, N):
    FN = 0
    for i in range(len(y)):
        if y[i] == T and y_predict[i] == N:
            FN += 1

    return FN


def calulate_recall(TP, FN):
    return (TP) / (TP + FN)


def calulate_precision(TP, FP):
    return (TP) / (TP + FP)


def calulate_accuracy(y, y_predict):
    sum = len(y)
    t = np.sum(y == y_predict)
    return t / sum


def get_martix(y, y_predict, T, N):
    TP = 0
    for i in range(len(y)):
        if y[i] == T and y_predict[i] == T:
            TP += 1
    TN = 0
    for i in range(len(y)):
        if y[i] == N and y_predict[i] == N:
            TN += 1
    FP = 0
    for i in range(len(y)):
        if y[i] == N and y_predict[i] == T:
            FP += 1
    FN = 0
    for i in range(len(y)):
        if y[i] == T and y_predict[i] == N:
            FN += 1

    recall = (TP) / (TP + FN)
    precision = (TP) / (TP + FP)
    decall = (TN) / (TN + FN)

    return recall, decall, precision, TP, TN, FP, FN