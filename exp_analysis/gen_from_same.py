"""
  @Time : 2022/3/29 15:11 
  @Author : Ziqi Wang
  @File : gen_from_same.py 
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


def generate(designer, start, n=4):#, player=MarioJavaAgents.Runner, k=4):
    obs_buffer = RingQueue(n)
    generator = get_generator()
    simulator = MarioProxy()
    repairer = DivideConquerRepairer()
    latvecs = []
    for z in start:
        # latvec = np.random.rand(20).astype(np.float32)
        obs_buffer.push(np.array(z).astype(np.float32))
        latvecs.append(np.array(z).astype(np.float32))

    # designer = Designer('exp_data/main/both/actor.pth')
    for i in range(50):
        obs = np.concatenate(obs_buffer.to_list())
        a = designer.act(obs)
        latvecs.append(a)
        obs_buffer.push(a)
    segs = process_levels(generator(torch.tensor(np.array(latvecs)).view(-1, nz, 1, 1)), True)
    full_lvl = repairer.repair(level_sum(segs))
    s1, s2, s3 = n*28, (n+10)*28, (n+25)*28
    e1, e2, e3 = s1+112, s2+112, s3+112
    return full_lvl[:, s1:e1].to_img(None), full_lvl[:, s2:e2].to_img(None), full_lvl[:, s3:e3].to_img(None)
    # trace = simulator.simulate_long(full_lvl, player, k)['trace']
    # img  = full_lvl.to_img_with_trace(trace, save_path=None)
    # return img.subsurface((5*28*16, 0, 8*28*16, 14*16))

    # simlt_res = MarioProxy.get_seg_infos(simulator.simulate_long(full_lvl))
    # img = full_lvl.to_img(None)
    # pygame.draw.lines(img, 'black', False, [(x, y - 8) for x, y in trace], 3)
    # part1 = img.subsurface((0, 0, 16*28*4, 16*14))
    # part2 = img.subsurface((16*28*4, 0, 16*28*8, 16*14))
    # full = pygame.Surface((16*28*12 + 8, 16*14))
    # full.fill('white')
    # full.blit(part1, (0, 0))
    # full.blit(part2, (16*28*4 + 8, 0))
    # return full

    # return full_lvl, full_trace


if __name__ == '__main__':
    with open(get_path('exp_analysis/extremum_lvls/g_latvec.json'), 'r') as f:
        start_point = json.load(f)
    # res = generate(Designer('exp_data/main/collector/actor.pth'), start_point, MarioJavaAgents.Collector, 10.)
    # pygame.image.save(res, get_path('exp_data/main/same_start/g/collector.png'))
    # res = generate(Designer('exp_data/main/killer/actor.pth'), start_point, MarioJavaAgents.Killer, 10.)
    # pygame.image.save(res, get_path('exp_data/main/same_start/g/killer.png'))
    # res = generate(Designer('exp_data/main/both/actor.pth'), start_point)
    # pygame.image.save(res, get_path('exp_data/main/same_start/g/both.png'))
    # res = generate(Designer('exp_data/main/content/actor.pth'), start_point)
    # pygame.image.save(res, get_path('exp_data/main/same_start/g/content.png'))
    # res = generate(Designer('exp_data/main/behaviour/actor.pth'), start_point)
    # pygame.image.save(res, get_path('exp_data/main/same_start/g/behaviour.png'))


    # level, trace = generate(Designer('exp_data/main/both/actor.pth'), start_point)
    # level.to_img('exp_data/main/same_start1/both.png')
    # level.save_img_with_trace(trace, 'exp_data/main/same_start/both_with_trace.png')
    #
    # level, trace = generate(Designer('exp_data/main/content/actor.pth'), start_point)
    # level.to_img('exp_data/main/same_start1/content.png')
    # level.save_img_with_trace(trace, 'exp_data/main/same_start/content_with_trace.png')

    # level, trace = generate(Designer('exp_data/main/behaviour/actor.pth'), start_point)
    # img = level.to_img(None)
    # pygame.draw.lines(img, 'black', False, [(x, y - 8) for x, y in trace], 3)
    # part1 = img.subsurface((0, 0, 16*28*4, 16*28))
    # part2 = img.subsurface((16*28*4, 0, 16*28*8, 16*28))
    # full = pygame.Surface((16*28*12 + 8, 16*28))
    # full.blit(part1, (0, 0))
    # full.blit(part2, (16*28*4 + 8, 0))
    # pygame.image.save(img, 'exp_data/main/same_start/behaviour.png')

    # pygame.draw.line(img, 'exp_data/main/same_start/behaviour.png')

    # pygame.image.save(img, 'exp_data/main/same_start/behaviour.png')
    # level.save_img_with_trace(trace, 'exp_data/main/same_start/behaviour_with_trace.png')
