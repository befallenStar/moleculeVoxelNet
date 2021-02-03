# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vn.loader.pointcloud2voxel import *


# conv3d + activation
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation='relu'):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'relu':
            return F.relu(x, inplace=True)
        elif self.activation == 'tanh':
            return torch.tanh(x)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout, activation='relu'):
        super(FCN, self).__init__()
        self.cin = cin
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.activation = activation

    def forward(self, x):
        # KK is the stacked k across batch
        x = self.linear(x.view(-1))
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)


class WaveNet(nn.Module):
    def __init__(self, cout):
        super(WaveNet, self).__init__()
        self.conv3d_1 = Conv3d(10, 64, 3, 2, 1, 'tanh')
        self.conv3d_2 = Conv3d(64, 128, 3, 2, 1, 'tanh')
        self.conv3d_3 = Conv3d(128, 256, 3, 2, 1, 'tanh')
        self.conv3d_4 = Conv3d(256, 512, 3, 2, 1, 'tanh')
        self.conv3d_5 = Conv3d(512, 128, 1, 1, 0, 'tanh')
        self.fcn_1 = FCN(3072, 1024, 'tanh')
        self.fcn_2 = FCN(1024, 384, 'tanh')
        self.fcn_3 = FCN(384, cout, 'tanh')

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        x = self.conv3d_5(x)
        x = self.fcn_1(x)
        x = self.fcn_2(x)
        x = self.fcn_3(x)
        return x

