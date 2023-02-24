"""
  @Time : 2022/1/25 14:36 
  @Author : Ziqi Wang
  @File : gan_train.py 
"""

import os
import glob
import time
import random
import numpy as np
from torch.optim import RMSprop
from itertools import product
from src.gan.gan_models import *
from src.gan.gan_use import process_levels
from src.utils.datastruct import batched_iter
from root import PRJROOT


def get_gan_train_data():
    H, W = MarioLevel.height, MarioLevel.default_seg_width
    path = 'levels/train'
    data = []
    for fpath in glob.glob(f'{path}/*.smblv'):
        num_lvl = MarioLevel.from_file(fpath).to_num_arr()
        _, length = num_lvl.shape
        for s in range(length - W):
            segment = num_lvl[:, s: s+W]
            if 6 in segment[:, 0] or 7 in segment[:, -1]:
                continue
            onehot = np.zeros([MarioLevel.num_tile_types, H, W])
            xs = [segment[i,j] for i, j in product(range(H), range(W))]
            ys = [k // W for k in range(H * W)]
            zs = [k % W for k in range(H * W)]
            onehot[xs, ys, zs] = 1
            data.append(onehot)
    return data

def set_parser(parser):
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--niter', type=int, default=2000, help='Number of epochs to train for')
    parser.add_argument('--eval_itv', type=int, default=10, help='Interval (in unit of iteration) of evaluate GAN and print log')
    parser.add_argument('--repeatG', type=int, default=5, help='Repeatly train G for how many time in one iteration')
    parser.add_argument('--repeatD', type=int, default=1, help='Repeatly train D for how many time in one iteration')
    parser.add_argument('--lrD', type=float, default=3e-4, help='Learning rate for Discriminator, default=3e-4')
    parser.add_argument('--lrG', type=float, default=3e-4, help='Learning rate for Generator, default=3e-4')
    parser.add_argument('--gpuid', type=int, default=-1, help='ID of gpu. If smaller than 0, use cpu')
    parser.add_argument('--res_path', type=str, default='GAN', help='Path related to \'/exp_data\' to store training data')
    parser.add_argument('--weight_clip', type=float, default=0.01, help='Clip the weight of discriminators within [-weight_clip, weight_clip]')

def train_gan(cfgs):
    device = 'cpu' if cfgs.gpuid < 0 else f'cuda:{cfgs.gpuid}'
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    optG = RMSprop(netG.parameters(), lr=cfgs.lrG)
    optD = RMSprop(netD.parameters(), lr=cfgs.lrD)
    data = get_gan_train_data()
    data = [torch.tensor(item, device=device, dtype=torch.float) for item in data]
    res_path = 'exp_data/' + cfgs.res_path
    os.makedirs(PRJROOT + res_path, exist_ok=True)
    start_time = time.time()
    for t in range(cfgs.niter):
        random.shuffle(data)
        for item, n in batched_iter(data, cfgs.batch_size):
            real = torch.stack(item)
            # Train
            for _ in range(cfgs.repeatD):
                for p in netD.parameters():
                    p.data.clamp_(-cfgs.weight_clip, cfgs.weight_clip)
                with torch.no_grad():
                    # z = torch.rand(n_init, nz, 1, 1, device=device) * 2 - 1
                    z = torch.randn(n, nz, 1, 1, device=device)
                    fake = netG(z)
                l_real = -netD(real).mean()
                l_fake = netD(fake).mean()

                optD.zero_grad()
                l_real.backward()
                l_fake.backward()
                optD.step()

            for _ in range(cfgs.repeatG):
                # z = torch.rand(n_init, nz, 1, 1, device=device) * 2 - 1
                z = torch.randn(n, nz, 1, 1, device=device)
                fake = netG(z)
                optG.zero_grad()
                loss_G = -netD(fake).mean()
                loss_G.backward()
                optG.step()
        # Evaluate
        if t % cfgs.eval_itv == (cfgs.eval_itv - 1):
            netG.eval()
            netD.eval()
            with torch.no_grad():
                real = torch.stack(data[:min(100, len(data))])
                z = torch.rand(100, nz, 1, 1, device=device) * 2 - 1
                # z = torch.randn(100, nz, 1, 1, device=device)
                fake = netG(z)
                y_real = netD(real).mean()
                y_fake = netD(fake).mean()
            print(
                'Iteration %d, y-real=%.3g, y-fake=%.3g, time: %.1fs' %
                (t+1, y_real, y_fake, time.time() - start_time)
            )
            netD.train()
            netG.train()
        if t % 50 == 49:
            netG.eval()
            netD.eval()
            with torch.no_grad():
                z = torch.rand(100, nz, 1, 1, device=device) * 2 - 1
                # z = torch.randn(100, nz, 1, 1, device=device)
                fake = netG(z)
            levels = process_levels(fake[:32], True)
            iteration_path = res_path + f'/iteration{t+1}'
            os.makedirs(iteration_path, exist_ok=True)
            for i, lvl in enumerate(levels):
                lvl.save(iteration_path + f'/sample{i}')
                lvl.to_img(iteration_path + f'/sample{i}.png')
            torch.save(netG, PRJROOT + iteration_path + '/generator.pth')
            netD.train()
            netG.train()
            pass


