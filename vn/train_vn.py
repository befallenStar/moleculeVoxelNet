# -*- encoding: utf-8 -*-
from time import time, strftime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange

from loader.data_loading import load_db
from loader.pointcloud2voxel import load_voxel
from voxelnet import SVFE
from wavenet import WaveNet

feature_length = 15


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        init.kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()


def train(net='voxel', epochs=5, scale='medium', size=None):
    if size is None:
        size = [36, 36, 36]
    try:
        # load atoms of ase formatter from ase.db
        # qm9 = load_db(path='ase_data', ase_db='qm9.db')
        qm9, propertieses, datas = load_db(path='ase_data',
                                           ase_db='qm9_{}_{}.db'.format(
                                               feature_length, scale))

        D, H, W = size

        # create a VoxelNet object
        if net == 'voxel':
            net = SVFE(feature_length)
            learning_rate = 0.03
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        elif net == 'wave':
            net = WaveNet(feature_length)
            learning_rate = 0.001
            optimizer = optim.Adadelta(net.parameters(), lr=learning_rate)
        # initialization
        net.apply(weights_init)
        # record the time cost
        losses = []

        # init the loss function
        loss_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        # learning rate should in [0.3, 0.1, 0.03]
        # for atoms, properties,data in zip(qm9, propertieses,datas):
        for epoch in range(epochs):
            start = time()
            w_loss, w_mae = 0, 1
            net.train()
            for i in trange(len(qm9)):
                atoms = qm9[i]
                properties = propertieses[i]
                # data = datas[i]
                # iterate the atoms
                voxel = load_voxel(atoms, D, H, W, sigma=2)
                # print("voxel: " + str(voxel.shape))
                # wrapper to variable
                voxel_features = Variable(torch.FloatTensor(voxel))

                feature = [value for value in properties.values()]
                # feature.extend([value for value in data.values()])
                feature = np.arctan(feature) * 2 / np.pi
                # feature = (feature - np.mean(feature)) / np.std(feature)
                feature = Variable(torch.FloatTensor(feature))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                vwfs = net(voxel_features)
                feature = feature.expand(vwfs.shape)
                # calculate loss
                loss = loss_fn(vwfs, feature)
                mae = MAE_fn(vwfs, feature)

                # backward
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()

                w_loss += loss.data
                w_mae += mae.data
                losses.append(loss.data)

            w_mae /= len(qm9) + 1

            print(
                "Epoch {:2d}, \tloss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                    epoch, w_loss, w_mae, time() - start))
            # losses.append(w_loss)
        torch.save(net, './model/voxelnet_{}.pth'.format(epochs))
        # plot the loss curve
        plt.plot(range(len(losses)), losses, linestyle='-')
        plt.savefig(r'./img/voxelnet_{}_{:.4f}.png'.format(strftime('%Y-%m-%d'),
                                                           w_mae))
        plt.show()
        # save the model
        # torch.save(net,'model/aseVoxelNet-1.pkl')
    except ValueError as e:
        print(str(e))


if __name__ == '__main__':
    # train(net='wave', epochs=10, scale='medium')
    train(net='wave', epochs=1, scale='medium', size=[36,36,36])
