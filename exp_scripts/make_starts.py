"""
  @Time : 2022/3/27 21:23 
  @Author : Ziqi Wang
  @File : make_starts.py 
"""
import sys

import numpy as np
import torch

from src.environment.reward_function import FunContent, FunBehaviour

sys.path.append('../')

import json
import os
import argparse
import time
from smb import MarioProxy, level_sum
from src.gan.gan_use import *
from src.utils.parallel import AsyncPoolWithLoad, TaskLoad
from src.utils.datastruct import batched_iter
from src.repairer.repairer import DivideConquerRepairer
from src.analyze.genlvl_statistics import generate_levels
from src.utils.datastruct import ConditionalDataBuffer



def dominate(a, b):
    return all(u <= v for u, v in zip(a, b)) and any(u < v for u, v in zip(a, b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str, default='models/generator.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_init', type=int, default=4)
    parser.add_argument('--space_size', type=float, default=1.)
    # parser.add_argument('--parallel', type=int, default=6)
    # parser.add_argument('--res_path', type=str, default='exp_data/starts100.json')

    args = parser.parse_args()
    generator = get_generator(args.generator_path, args.device)

    # eval_pool = AsyncPoolWithLoad(args.parallel)
    W = MarioLevel.default_seg_width
    simulator = MarioProxy()
    repairer = DivideConquerRepairer()
    fun_c_func, fun_b_func = FunContent(), FunBehaviour()

    buffer = []
    # archive = ConditionalDataBuffer()
    n_playable, n_samples = 0, 0
    while n_playable < args.n_samples:
        z = sample_latvec(args.n, device=args.device)
        segs = process_levels(generator(z), True)
        # level = level_sum(segs)
        # fc, fb = evaluate(level_sum(segs))
        repaired = repairer.repair(level_sum(segs))
        simlt = simulator.simulate_long(repaired)
        if not simlt['restarts']:
            simlt_res = MarioProxy.get_seg_infos(simlt)
            fc = np.array(
                fun_c_func.compute_rewards(
                    segs=[repaired[:, s:s + W] for s in range(0, repaired.w, W)]
                )
            ).mean()
            fb = np.array(
                fun_b_func.compute_rewards(
                    segs=[repaired[:, s:s + W] for s in range(0, repaired.w, W)],
                    simlt_res=simlt_res
                )
            ).mean()
            buffer.append({
                'z': [item.squeeze().cpu().tolist() for item in z],
                'fc': fc, 'fb': fb
            })
            n_playable += 1
        n_samples += 1
        print(f'{n_playable}/{args.n_samples}-{n_samples}')

    non_dominated = []
    for item in buffer:
        fc, fb = item['fc'], item['fb']
        if all((not dominate((x['fc'], x['fb']), (fc, fb))) for x in buffer):
            non_dominated.append(item)
    with open(get_path(f'exp_data/starts{args.n_samples}.json'), 'w') as f:
        json.dump(non_dominated, f)

