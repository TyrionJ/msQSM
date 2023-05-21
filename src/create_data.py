import numpy as np
import nibabel as nb
import os
import h5py
import torch
from scipy.io import loadmat
from torch import from_numpy as to_torch

from config import train_file, valid_file
from utils.TKD import cal_QSM0

"input data folder (samples)"
data_root = '../data/in-house'

gyro = 2 * np.pi * 42.5857
VALID = 16 ** 3
"""
cropped into patches with 50% overlap
"""
x_patch = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224), (192, 256)]
y_patch = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224), (192, 256)]
z_patch = {
    128: [(0, 64), (32, 96), (64, 128)],
    136: [(0, 64), (36, 100), (72, 136)],
    138: [(0, 64), (37, 101), (74, 138)]
}

"test group (samples)"
test_group = ['Subject002']


def make_patch(mix, fld, mask):
    mix_patch_arr, fld_patch_arr, msk_patch_arr = [], [], []

    for xx in x_patch:
        for yy in y_patch:
            for zz in z_patch[mix.shape[-1]]:
                msk_patch = mask[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                if msk_patch.sum() < VALID:
                    continue
                mix_patch = mix[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]
                fld_patch = fld[xx[0]:xx[1], yy[0]:yy[1], zz[0]:zz[1]]

                msk_patch_arr.append(msk_patch)
                fld_patch_arr.append(fld_patch)
                mix_patch_arr.append(mix_patch)

    return mix_patch_arr, fld_patch_arr, msk_patch_arr


def create_dataset():
    """
    % Inputs:
    %   TissuePhase : Result of 3D V-SHARP in STISuite_V3.0 toolbox
    %   Mask : Result of 3D V-SHARP
    %   TE : unit s
    %   B0 : unit T
    """
    people = sorted(os.listdir(data_root))
    tr_qsm0_patches, vd_qsm0_patches = [], []
    tr_fld_patches, vd_fld_patches = [], []
    tr_msk_patches, vd_msk_patches = [], []

    for i, person in enumerate(people):
        print(f'\r[{i + 1}/{len(people)}] Processing {person} ...', end='')

        mask = nb.load(f'{data_root}/{person}/mask.nii.gz')
        phs = nb.load(f'{data_root}/{person}/tissue_phase.nii.gz')
        header = loadmat(f'{data_root}/{person}/header.mat')

        ori_vec = header['B0_vector'][0]
        voxel_size = header['voxelsize'][0]
        TE = np.mean(header['TEs'])

        mask = mask.get_fdata()
        fld = phs.get_fdata() / (TE * header['B0'][0, 0] * gyro)

        QSM0 = cal_QSM0(to_torch(fld).float(), ori_vec, voxel_size, to_torch(mask)).numpy()
        qsm0_patch_arr, fld_patch_arr, msk_patch_arr = make_patch(QSM0, fld, mask)

        if person in test_group:
            vd_qsm0_patches += qsm0_patch_arr
            vd_fld_patches += fld_patch_arr
            vd_msk_patches += msk_patch_arr
        else:
            tr_qsm0_patches += qsm0_patch_arr
            tr_fld_patches += fld_patch_arr
            tr_msk_patches += msk_patch_arr
    print()

    ks = ['qsm0', 'fld', 'msk']

    print(f'Writing valid_file: size={len(vd_qsm0_patches)}')
    f = h5py.File(valid_file, 'w')
    ds = [vd_qsm0_patches, vd_fld_patches, vd_msk_patches]
    for i in range(len(ds)):
        f.create_dataset(ks[i], data=np.array(ds[i]))
    f.close()

    print(f'Writing train_file: size={len(tr_qsm0_patches)}')
    f = h5py.File(train_file, 'w')
    ds = [tr_qsm0_patches, tr_fld_patches, tr_msk_patches]
    for i in range(len(ds)):
        f.create_dataset(ks[i], data=np.array(ds[i]))
    f.close()


if __name__ == '__main__':
    create_dataset()
