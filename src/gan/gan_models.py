"""
  @Time : 2022/1/24 14:40 
  @Author : Ziqi Wang
  @File : gan_models.py 
"""
import torch
from torch import nn
from src.gan.gan_config import nz
from smb import MarioLevel


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 256, (4, 4), bias=False),
            nn.ReLU(True), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.ReLU(True), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ReLU(True), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, MarioLevel.num_tile_types, (3, 4), (1, 2), (1, 1), bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(MarioLevel.num_tile_types, 64, (3, 4), (1, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, (4, 4), bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    noise = torch.rand(2, nz, 1, 1) * 2 - 1
    netg = Generator()
    x = netg(noise)
    print(x.size())
    netd = Discriminator()
    y = netd(x)
    print(y.size())
