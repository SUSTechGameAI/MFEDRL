"""
  @Time : 2021/11/26 14:34 
  @Author : Ziqi Wang
  @File : replay_memory.py
"""

import random

import numpy as np
import torch

from src.utils.datastruct import RingQueue


class ReplayMem:
    def __init__(self, capacity=int(5e4)):
        self.queue = RingQueue(capacity)

    def add(self, o, a, r, op, d):
        self.queue.push((o, a, r, op, d))

    def add_batched(self, obs, actions, rewards, next_obs, dones):
        for o, a, r, op, d in zip(obs, actions, rewards, next_obs, dones):
            self.add(o, a, r, op, d)

    def sample(self, n, device='cpu'):
        tuples = random.sample(self.queue.main, n)
        o, a, r, op, d = [], [], [], [], []
        for item in tuples:
            o.append(item[0])
            a.append(item[1])
            r.append(item[2])
            op.append(item[3])
            d.append(1 if item[4] else 0)
        o = torch.tensor(np.array(o), device=device)
        a = torch.tensor(np.array(a), device=device)
        r = torch.tensor(np.array(r), device=device)
        op = torch.tensor(np.array(op), device=device)
        d = torch.tensor(np.array(d), device=device)
        return o, a, r, op, d

    def __len__(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()