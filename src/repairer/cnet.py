"""
  @Time : 2022/1/5 14:09 
  @Author : Ziqi Wang
  @File : cnet.py 
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch import nn
from src.utils.filesys import get_path
from root import PRJROOT
from smb import MarioLevel


class CNet(nn.Module):
    def __init__(self, n_tile_types, emb_dim=6, layers=(32, 32)):
        super(CNet, self).__init__()
        tmp = [emb_dim * 8, *layers]
        after_emb = []
        for i in range(len(layers)):
            after_emb.append(nn.Linear(tmp[i], tmp[i+1]))
            after_emb.append(nn.ReLU())
        self.main = nn.Sequential(
            nn.Embedding(n_tile_types+1, emb_dim),
            nn.Flatten(),
            *after_emb,
            nn.Linear(tmp[-1], n_tile_types),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    pass


def get_data():
    with open(PRJROOT + 'levels/train/cnet_data.json', 'r') as f:
        data = json.load(f)
    x = torch.tensor([item[0] for item in data])
    y = torch.tensor([item[1] for item in data])
    return TensorDataset(x, y)
    pass

def set_parser(parser):
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning ratec, default=2e-4')
    parser.add_argument('--save_path', type=str, default='models', help='Path related to \'/exp_data\' to store training data')

def train_cnet(cfgs):
    model = CNet(MarioLevel.num_tile_types)
    dataset = get_data()
    dataloader = DataLoader(dataset, cfgs.batch_size, True)
    optimizer = Adam(model.parameters(), lr=cfgs.lr, weight_decay=5e-4)
    n = len(dataset)
    for t in range(cfgs.niter):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            y_tilde = model(x)
            loss = F.binary_cross_entropy(y_tilde, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        if t % 100 == 99:
            print('iteration %d, loss: %.4g' % (t+1, total_loss / n))
    torch.save(model, get_path(cfgs.save_path) + '/Cnet.pth')
    torch.save(model.state_dict(), get_path(cfgs.save_path) + '/Cnet_state_dict.pth')
    pass

def get_cnet(path='models/Cnet.pth', device='cpu'):
    safe_path = get_path(path)
    return torch.load(safe_path, map_location=device)

