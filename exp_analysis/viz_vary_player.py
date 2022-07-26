"""
  @Time : 2022/4/5 20:35 
  @Author : Ziqi Wang
  @File : viz_vary_player.py 
"""
import json

import numpy as np
import torch
import pygame
from smb import MarioProxy, level_sum, MarioJavaAgents
from src.designer.use_designer import Designer
from src.gan.gan_config import nz
from src.gan.gan_use import get_generator, process_levels
from src.repairer.repairer import DivideConquerRepairer
from src.utils.datastruct import RingQueue
from src.utils.filesys import get_path


def generate(designer, start, player=''):
    obs_buffer = RingQueue(4)
    generator = get_generator()
    simulator = MarioProxy()
    repairer = DivideConquerRepairer()
    latvecs = []
    for z in start:
        obs_buffer.push(np.array(z).astype(np.float32))
        latvecs.append(np.array(z).astype(np.float32))

    for i in range(8):
        obs = np.concatenate(obs_buffer.to_list())
        a = designer.act(obs)
        latvecs.append(a)
        obs_buffer.push(a)
    segs = process_levels(generator(torch.tensor(np.array(latvecs)).view(-1, nz, 1, 1)), True)
    full_lvl = repairer.repair(level_sum(segs))
    trace_r = simulator.simulate_long(full_lvl, MarioJavaAgents.Runner)['trace']
    trace_k = simulator.simulate_long(full_lvl, MarioJavaAgents.Killer, 10.)['trace']
    trace_c = simulator.simulate_long(full_lvl, MarioJavaAgents.Collector, 10.)['trace']

    img = full_lvl.to_img(None)
    if player == 'runner':
        pygame.draw.lines(img, 'white', False, [(x, y - 8) for x, y in trace_k], 2)
        pygame.draw.lines(img, 'red', False, [(x, y - 8) for x, y in trace_c], 2)
        pygame.draw.lines(img, 'black', False, [(x, y - 8) for x, y in trace_r], 3)
    elif player == 'killer':
        pygame.draw.lines(img, 'red', False, [(x, y - 8) for x, y in trace_c], 2)
        pygame.draw.lines(img, 'black', False, [(x, y - 8) for x, y in trace_r], 2)
        pygame.draw.lines(img, 'white', False, [(x, y - 8) for x, y in trace_k], 3)
    elif player == 'collector':
        pygame.draw.lines(img, 'black', False, [(x, y - 8) for x, y in trace_r], 3)
        pygame.draw.lines(img, 'white', False, [(x, y - 8) for x, y in trace_k], 3)
        pygame.draw.lines(img, 'red', False, [(x, y - 8) for x, y in trace_c], 3)
    else:
        pygame.draw.lines(img, 'white', False, [(x, y - 8) for x, y in trace_k], 3)
        pygame.draw.lines(img, 'red', False, [(x, y - 8) for x, y in trace_c], 3)
        pygame.draw.lines(img, 'black', False, [(x, y - 8) for x, y in trace_r], 3)
    return img.subsurface((5*28*16, 0, 8*28*16, 14*16))




if __name__ == '__main__':
    # start_point = (np.random.rand(4, nz) * 2 - 1).astype(np.float32)
    with open(get_path('exp_analysis/extremum_lvls/g_latvec.json'), 'r') as f:
        start_point = json.load(f)

    res = generate(Designer('exp_data/main/both/actor.pth'), start_point)
    pygame.image.save(res, get_path('exp_data/main/same_start/g/both.png'))
    res = generate(Designer('exp_data/main/content/actor.pth'), start_point)
    pygame.image.save(res, get_path('exp_data/main/same_start/g/content.png'))
    res = generate(Designer('exp_data/main/behaviour/actor.pth'), start_point)
    pygame.image.save(res, get_path('exp_data/main/same_start/g/behaviour.png'))
    # res = generate(Designer('exp_data/main/killer/actor.pth'), start_point, 'killer')
    # pygame.image.save(res, get_path('exp_data/main/same_start/rand/killer.png'))
    # res = generate(Designer('exp_data/main/collector/actor.pth'), start_point, 'collector')
    # pygame.image.save(res, get_path('exp_data/main/same_start/rand/collector.png'))

