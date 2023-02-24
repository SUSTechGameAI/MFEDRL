"""
  @Time : 2022/9/30 13:52 
  @Author : Ziqi Wang
  @File : generate.py
"""
import torch
import random
import numpy as np
from src.gan.gan_use import *
from src.gan.gan_config import nz
from smb import level_sum, save_batch
from src.utils.filesys import get_path
from src.utils.datastruct import RingQueue
from src.environment.env_cfgs import history_len
from src.designer.use_designer import Designer
from src.repairer.repairer import DivideConquerRepairer


def generate_levels(designer, dest_path='', n=1, h=25, parallel=1, repair=False):
    levels = []
    latvecs = []
    obs_queues = [RingQueue(history_len) for _ in range(parallel)]
    # init_arxv = np.load(get_path('smb/init_latvecs.npy'))
    generator = get_generator(device='cuda:0')
    repairer = DivideConquerRepairer()
    while len(levels) < n:
        # segbf = [[] for _ in range(parallel)]
        veclists = [[] for _ in range(parallel)]
        for queue, veclist in zip(obs_queues, veclists):
            queue.clear()
            init_latvecs = sample_latvec(history_len).squeeze().numpy()
            # init_latvec = init_arxv[random.randrange(0, len(init_arxv))]
            for latvec in init_latvecs:
                queue.push(latvec)
                veclist.append(latvec)
        for _ in range(h):
            obs = np.stack([np.concatenate(queue.to_list()) for queue in obs_queues])
            actions = designer.act(obs)
            for queue, veclist, action in zip(obs_queues, veclists, actions):
                queue.push(action)
                veclist.append(action)
        for veclist in veclists:
            latvecs.append(np.stack(veclist))
            z = torch.tensor(latvecs[-1], device='cuda:0').view(-1, nz, 1, 1)
            lvl = level_sum(process_levels(generator(z), True))
            if repair:
                lvl = repairer.repair(lvl)
            levels.append(lvl)
        #     for seglist, seg, queue, veclist, action in zip(segbf, segs, obs_queues, veclists, actions):
        #         seglist.append(seg)
        #         queue.push(action)
        #         veclist.append(action)
        # levels += [lvlhcat(seglist) for seglist in segbf]
        # latvecs += [np.stack(veclist) for veclist in veclists]
        # print(f'{len(levels)}/{n} generated')
    if dest_path:
        save_batch(levels[:n], dest_path)
        np.save(get_path(dest_path), np.stack(latvecs[:n]))
    return levels[:n], latvecs[:n]


if __name__ == '__main__':
    # generate_levels(
    #     Designer('exp_data/recurrent_transition/n4/gm070/actor.pth'),
    #     'exp_analysis/endless_gen/data/gm070n4',
    #     100, 50, 10
    # )
    generate_levels(
        Designer('exp_data/recurrent_transition/n4/gm080/actor.pth'),
        'exp_analysis/endless_gen/data/gm080n4',
        100, 50, 10
    )
    generate_levels(
        Designer('exp_data/recurrent_transition/n4/gm090/actor.pth'),
        'exp_analysis/endless_gen/data/gm090n4',
        100, 50, 10
    )
    generate_levels(
        Designer('exp_data/recurrent_transition/n4/gm099/actor.pth'),
        'exp_analysis/endless_gen/data/gm099n4',
        100, 50, 10
    )

