import torch
import torch.nn as nn
import torch.nn.functional as F


# conv1d + bn + relu
class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=False):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=False):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        # self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        H, W, D, _ = x.shape
        x = self.linear(x.reshape(H * W * D, -1))
        x = F.relu(x)
        return x.reshape(H, W, D, -1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x):
        # point-wise feauture
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 3)[0].unsqueeze(3).repeat(1, 1, 1, self.units)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=3)
        # apply mask
        # mask = mask.unsqueeze(3)
        # mask = mask.repeat(1, 1, 1, self.units * 2)
        # pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self, cout):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(10, 32)
        self.vfe_2 = VFE(32, 64)
        self.vfe_3 = VFE(64, 128)
        self.fcn = FCN(64, 64)
        self.conv2d_1 = Conv2d(64, 2 * cout, 3, 2, 1)
        self.conv2d_2 = Conv2d(2 * cout, cout, 3, 2, 1)

    def forward(self, x):
        # mask = torch.ne(torch.max(x, 3)[0], 0)
        x = self.vfe_1(x)
        # mask = torch.ne(torch.max(x, 3)[0], 0)
        x = self.vfe_2(x)
        # mask = torch.ne(torch.max(x, 3)[0], 0)
        # x=self.vfe_3(x)
        x = self.fcn(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = x.permute(0, 2, 3, 1)
        return x
