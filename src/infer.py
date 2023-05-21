import argparse
from scipy.io import loadmat
import torch
import nibabel as nb
import numpy as np
import time

from utils import set_env, load_model
from config import d_type, pretrained_model
from utils.TKD import cal_QSM0

data_root = '../data'

gamma = 42.5857
gyro = 2 * np.pi * gamma


def load_hemorrhage():
    dr = f'{data_root}/Hemorrhage'
    mat = loadmat(f'{dr}/data.mat')

    tissue = mat['tissue_phase']
    mask = mat['mask']

    field = tissue / (mat['TE'][0, 0] * 1e-3 * mat['B0'][0, 0] * gyro)
    nii_file = f'{dr}/msQSM.nii.gz'

    return nii_file, field, mask, mat['B0_vector'][0], mat['voxel_size'][0]


def load_Berkeley_Mouse():
    dr = f'{data_root}/MouseData'

    phase = nb.load(f'{dr}/tissue_phase.nii.gz')
    mask = nb.load(f'{dr}/new_mask.nii.gz')
    P = loadmat(f'{dr}/Parameters.mat')

    TE = P['TE1'][0, 0] + P['delta_TE'][0, 0]
    field = phase.get_fdata() / (TE * 1e-3 * P['B0'][0, 0] * gyro)

    nii_file = f'{data_root}/MouseData/msQSM.nii.gz'

    return nii_file, field, mask.get_fdata(), P['B0_vector'][0], P['voxelsize'][0]


def load_inhouse(sub):
    dr = f'{data_root}/in-house'
    tissue = nb.load(rf'{dr}/{sub}/tissue_phase.nii.gz')
    mask = nb.load(f'{dr}/{sub}/mask.nii.gz')
    header = loadmat(f'{dr}/{sub}/header.mat')

    mask = mask.get_fdata()
    field = tissue.get_fdata() / ((np.average(header['TEs'])) * header['B0'][0, 0] * gyro)

    nii_file = f'{dr}//{sub}/msQSM.nii.gz'
    return nii_file, field, mask, header['B0_vector'][0], header['voxelsize'][0]


def test(model, field, mask, ori_vec, voxel_size, device):
    field = torch.from_numpy(field.astype(np.float32)).to(device)
    mask = mask.to(device)
    t1 = time.process_time()

    QSM0 = cal_QSM0(field, ori_vec, voxel_size, mask)
    X = QSM0.unsqueeze(dim=0).unsqueeze(dim=0)

    t2 = time.process_time()
    X = X.to(dtype=d_type)

    model.eval()
    with torch.no_grad():
        t3 = time.process_time()
        Y = model(X)
        t4 = time.process_time()

    print(f'  duration: {t2 - t1 + t4 - t3}')

    return (Y[0, 0] * mask).cpu(), QSM0.cpu()


def test_and_save(model, field, mask, ori_vec, voxel_sz, device, nii_file):
    rst, MIX = test(model, field, torch.from_numpy(mask), ori_vec, voxel_sz, device)
    if nii_file is not None:
        nb.Nifti1Image(rst.numpy(), __get_affine(voxel_sz)).to_filename(nii_file)


def main(t_type, d):
    print(f'-------------------------------\nProcessing {t_type} ...')
    device = set_env(d)
    _, model, _, _, _ = load_model(device, pretrained_model)

    if t_type == 'in-house':
        nii_file, field, mask, ori_vec, voxel_sz = load_inhouse(sub='Subject001')
        test_and_save(model, field, mask, ori_vec, voxel_sz, device, nii_file)

    if t_type == 'Berkeley_Mouse':
        nii_file, field, mask, ori_vec, voxel_sz = load_Berkeley_Mouse()
        test_and_save(model, field, mask, ori_vec, voxel_sz, device, nii_file)

    if t_type == 'hemorrhage':
        nii_file, field, mask, ori_vec, voxel_sz = load_hemorrhage()
        test_and_save(model, field, mask, ori_vec, voxel_sz, device, nii_file)


def __get_affine(vx_sz):
    return [[vx_sz[0], 0, 0, 0], [0, vx_sz[1], 0, 0], [0, 0, vx_sz[2], 0], [0, 0, 0, 1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRI DIPOLE INVERSION')
    parser.add_argument('-d', '--device', default='0', type=str)
    parser.add_argument('-g', '--grad', default=0.01, type=float, choices=[0, 0.01, 0.05, 0.1, 0.5])
    args = parser.parse_args()

    main('in-house', args.device)
    main('hemorrhage', args.device)
    main('Berkeley_Mouse', args.device)
