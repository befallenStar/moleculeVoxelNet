# -*- encoding: utf-8 -*-
from time import time, strftime

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader

from Alchemy_dataset import TencentAlchemyDataset, batcher
from sch import SchNetModel

feature_length = 15


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        init.kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()


def sch_train(epochs=80, device=torch.device('cpu')):
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(),
                                shuffle=False,
                                num_workers=0)

    model = SchNetModel(norm=True, output_dim=15)
    model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    losses = []
    for epoch in range(epochs):
        start = time()
        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(alchemy_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()
        w_mae /= idx + 1

        print(
            "Epoch {:2d}, \tloss: {:.7f}, \tmae: {:.7f}, \ttime: {:.3f}".format(
                epoch, w_loss, w_mae, time() - start))
        losses.append(w_loss)
    torch.save(model, r'./model/schnet-{}.pth'.format(epochs))
    plt.plot(range(epochs), losses, color='red')
    plt.savefig(
        r'./img/schnet_{}_{:.4f}.png'.format(strftime('%Y-%m-%d'), str(w_mae)))
    plt.show()


if __name__ == '__main__':
    sch_train(epochs=1)
