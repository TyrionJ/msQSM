import torch
from torch.fft import fftshift
import numpy as np
from multiprocessing import Pool


def create_kernel(shape, ori_vec=None, vox_sz=None, device=None):
    if vox_sz is None:
        vox_sz = [1., 1., 1.]
    if ori_vec is None:
        ori_vec = [0., 0., 1.]
    if device is None:
        device = torch.device('cpu')
    N = torch.tensor(shape, dtype=torch.int, device=device)
    kx, ky, kz = torch.meshgrid(torch.arange(-N[0] // 2, N[0] // 2, device=device),
                                torch.arange(-N[1] // 2, N[1] // 2, device=device),
                                torch.arange(-N[2] // 2, N[2] // 2, device=device))

    spatial = torch.tensor(vox_sz, dtype=torch.double)
    kx = (kx / kx.abs().max()) / spatial[0]
    ky = (ky / ky.abs().max()) / spatial[1]
    kz = (kz / kz.abs().max()) / spatial[2]
    k2 = kx ** 2 + ky ** 2 + kz ** 2 + 2.2204e-16

    tk = 1 / 3 - (kx * ori_vec[0] + ky * ori_vec[1] + kz * ori_vec[2]) ** 2 / k2
    tk = fftshift(tk)

    return tk


def orient_vector(x_deg, y_deg):
    rad_x = -x_deg * np.pi / 180
    rad_y = -y_deg * np.pi / 180

    return [np.sin(rad_y) * np.cos(rad_x), np.sin(rad_x), np.cos(rad_x) * np.cos(rad_y)]


def do_mapping(args):
    x_arr, y_arr, z_arr, data_arr, knl_mask, data, mask, local = args
    arr_size, _, _, Z = data_arr.shape
    mid_z = Z // 2

    for i in range(len(z_arr)):
        x, y, z = x_arr[i], y_arr[i], z_arr[i]

        if z < mid_z:
            data[:, x, y, mid_z - z - 1:mid_z] = data_arr[:, x, y, 0:z + 1]
            mask[x, y, mid_z - z - 1:mid_z] = knl_mask[x, y, 0:z + 1]
            local[x, y, 0] = z
        else:
            data[:, x, y, mid_z:Z - z + mid_z] = data_arr[:, x, y, z:Z]
            mask[x, y, mid_z:Z - z + mid_z] = knl_mask[x, y, z:Z]
            local[x, y, 1] = z

    return data, mask, local


def compress(data_arr, knl_mask):
    data_arr = torch.stack(data_arr)
    arr_size = len(data_arr)
    X, Y, Z = knl_mask.shape
    mid_z = Z // 2

    mask = torch.zeros(knl_mask.shape)
    data = torch.zeros_like(data_arr)
    local = torch.zeros([X, Y, 2], dtype=torch.short)

    surface = knl_mask.clone()
    surface[:, :, 0:mid_z - 1] = surface[:, :, 0:mid_z - 1] - surface[:, :, 1:mid_z]
    surface[:, :, mid_z + 1:Z] = surface[:, :, mid_z + 1:Z] - surface[:, :, mid_z:Z - 1]
    surface[surface < 0] = 0

    s_arr = torch.where(surface == 1)
    p_num = 12
    step = int(np.ceil(len(s_arr[0]) / p_num))
    pool = Pool(processes=p_num)
    ps = []

    for idx in range(0, p_num):
        x_arr = s_arr[0][idx * step:(idx + 1) * step]
        y_arr = s_arr[1][idx * step:(idx + 1) * step]
        z_arr = s_arr[2][idx * step:(idx + 1) * step]
        dat = torch.zeros_like(data_arr)
        msk = torch.zeros_like(knl_mask)
        loc = torch.zeros_like(local)
        ags = [x_arr, y_arr, z_arr, data_arr, knl_mask, dat, msk, loc]

        ps.append(pool.apply_async(do_mapping, (ags,)))

    pool.close()
    pool.join()
    for p in ps:
        dat, msk, loc = p.get()
        data += dat
        mask += msk
        local += loc

    min_z, max_z = 0, Z
    for z in range(0, Z):
        if mask[:, :, z].sum() > 0:
            min_z = z
            break
    for z in range(Z - 1, -1, -1):
        if mask[:, :, z].sum() > 0:
            max_z = z
            break

    info = (local, knl_mask.shape)
    data = tuple(data[i][:, :, min_z:max_z + 1] for i in range(arr_size))
    return data + (mask[:, :, min_z:max_z + 1], info)


def decompress(data_arr, compress_info):
    data_arr = torch.stack(data_arr)
    local_info, shape = compress_info
    n, X, Y, Z = data_arr.shape
    mid_z = Z // 2
    ZZ = shape[-1]

    o_data = torch.zeros((n,) + shape)

    for x in range(X):
        for y in range(Y):
            z1 = local_info[x, y, 0]
            z2 = local_info[x, y, 1]

            if z1 >= mid_z:
                o_data[:, x, y, z1 + 1 - mid_z:z1 + 1] = data_arr[:, x, y, 0:mid_z]
            else:
                o_data[:, x, y, 0:z1 + 1] = data_arr[:, x, y, mid_z - z1 - 1:mid_z]

            if ZZ - z2 >= mid_z:
                o_data[:, x, y, z2:z2 + mid_z] = data_arr[:, x, y, mid_z:]
            else:
                o_data[:, x, y, z2:ZZ] = data_arr[:, x, y, mid_z:mid_z + ZZ - z2]

    return tuple([o_data[i] for i in range(n)])
