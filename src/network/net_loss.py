import torch
import torch.nn as nn

from utils.morphology import im_morphology
from utils.normalization import mean_std


class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_hat, x, fld, msk):
        y_hat *= msk
        fld *= msk

        l1 = 10 * x.abs() * (y_hat - x)
        Z = torch.zeros_like(l1)
        l_cycle = self.l1_loss(l1[msk == 1], Z[msk == 1])

        f_mphx, f_mphy, f_mphz = im_morphology(fld)
        y_mphx, y_mphy, y_mphz = im_morphology(y_hat)

        f_mphx, f_mphy, f_mphz = f_mphx[msk == 1], f_mphy[msk == 1], f_mphz[msk == 1]
        y_mphx, y_mphy, y_mphz = y_mphx[msk == 1], y_mphy[msk == 1], y_mphz[msk == 1]

        f_mphx = abs(mean_std(f_mphx))
        f_mphy = abs(mean_std(f_mphy))
        f_mphz = abs(mean_std(f_mphz))

        y_mphx = abs(mean_std(y_mphx))
        y_mphy = abs(mean_std(y_mphy))
        y_mphz = abs(mean_std(y_mphz))

        l_grad = (self.l1_loss(y_mphx, f_mphx)
                  + self.l1_loss(y_mphy, f_mphy)
                  + self.l1_loss(y_mphz, f_mphz)) / 3

        return l_cycle, l_grad
