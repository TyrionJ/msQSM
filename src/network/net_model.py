from torch import nn


class NetModel(nn.Module):
    def __init__(self):
        super(NetModel, self).__init__()

        feats = 48
        self.in_convs = nn.Sequential(nn.Conv3d(1, feats//2, 3, 1, 1),
                                      nn.Conv3d(feats//2, feats, 3, 1, 1))
        self.blocks = nn.Sequential(ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats),
                                    ResBlock(feats, feats))
        self.out_convs = nn.Sequential(nn.Conv3d(feats, feats//2, 3, 1, 1),
                                       nn.Conv3d(feats//2, 1, 3, 1, 1))

    def forward(self, x):
        x = self.in_convs(x)
        x = self.blocks(x)
        x = self.out_convs(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm3d(out_channels)

        self.drop = nn.Dropout3d(p=0.2)
        self.actv = nn.GELU()

    def forward(self, x):
        _x = self.actv(self.norm1(self.conv1(x)))
        _x = self.drop(_x)
        _x = self.norm2(self.conv2(_x))

        return self.actv(_x + x)
