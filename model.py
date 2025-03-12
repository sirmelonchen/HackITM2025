import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(gc, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.lrelu(self.conv3(out))
        out = self.conv4(out) + x  # Residual Connection
        return out

class RRDB(nn.Module):
    def __init__(self, channels, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, gc)
        self.rdb2 = ResidualDenseBlock(channels, gc)
        self.rdb3 = ResidualDenseBlock(channels, gc)

    def forward(self, x):
        return self.rdb1(self.rdb2(self.rdb3(x))) + x  # Residual Connection

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        out = self.conv_first(x)
        out = self.rrdb_blocks(out)
        out = self.conv_last(out)
        return out
