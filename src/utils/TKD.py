from typing import Any

import numpy as np
import torch
from torch.fft import fftn, ifftn

from utils.morphology import im_morphology
from utils.normalization import min_max
from utils.kernel import create_kernel


def cal_kTKD(field, ori=None, thr=None, voxel_size=None):
    if voxel_size is None:
        voxel_size = [1., 1., 1.]
    if ori is None:
        ori = [0., 0., 1.]
    if thr is None:
        thr = 0.2

    kernel = create_kernel(field.shape, ori, voxel_size, field.device)
    ill = kernel.abs() < thr
    kernel[ill] = thr * torch.sign(kernel[ill])
    kernel[kernel == 0] = thr

    k_field = fftn(field)
    k_tkd = k_field / kernel

    return k_tkd


def cal_TKD(field, ori=None, thr=None, voxel_size=None, mask=None):
    kTKD = cal_kTKD(field, ori, thr, voxel_size)

    TKD = ifftn(kTKD)
    TKD = torch.abs(TKD) * torch.sign(torch.real(TKD))

    if mask is not None:
        TKD *= mask

    return TKD


def cal_QSM0(field, ori_vec, voxel_size, mask):
    sx, sy, sz = im_morphology(field.unsqueeze(dim=0).unsqueeze(dim=0))
    mag = abs(sx) + abs(sy) + abs(sz)
    mag = min_max(mag[0, 0])

    TKD1 = cal_TKD(field, ori_vec, 0.1, voxel_size, mask)
    TKD2 = cal_TKD(field, ori_vec, 0.2, voxel_size, mask)

    QSM0 = mag * TKD1 + (1 - mag) * TKD2
    QSM0 = nonlinear_process(QSM0, ori_vec, voxel_size, mask)

    return QSM0


def nonlinear_process(QSM0, ori_vec, voxel_size, mask):
    kernel = create_kernel(QSM0.shape, ori_vec, voxel_size, mask.device)
    pos = kernel.abs() < 0.1
    kQSM0 = fftn(QSM0)
    kQSM0[pos] *= torch.pow(abs(kernel[pos]), 0.2) * torch.sign(kernel[pos])
    QSM0 = ifftn(kQSM0)
    QSM0 = torch.abs(QSM0) * torch.sign(torch.real(QSM0)) * mask

    return QSM0
