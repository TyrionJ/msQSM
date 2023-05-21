import torch
from torch.nn.functional import conv3d

kx = torch.FloatTensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])
ky = torch.FloatTensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
kz = torch.FloatTensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])

wx = kx.unsqueeze(dim=0).unsqueeze(dim=0)
wy = ky.unsqueeze(dim=0).unsqueeze(dim=0)
wz = kz.unsqueeze(dim=0).unsqueeze(dim=0)


def im_morphology(data):
    global wx, wy, wz

    dx = conv3d(data, wx.to(data.device), padding=1)
    dy = conv3d(data, wy.to(data.device), padding=1)
    dz = conv3d(data, wz.to(data.device), padding=1)

    return dx, dy, dz
