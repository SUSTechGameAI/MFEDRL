"""
  @Time : 2022/3/22 18:26 
  @Author : Ziqi Wang
  @File : make_fun_path.py 
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from smb import MarioLevel, traverse_batched_level_files, traverse_level_files, level_sum, MarioProxy
from src.algo.ac_model import SquashedGaussianMLPActor
from src.environment.reward_function import FunContent, FunBehaviour
from src.gan.gan_config import nz
from src.gan.gan_use import process_levels, get_generator
from src.repairer.repairer import DivideConquerRepairer
from src.utils.datastruct import RingQueue
from src.utils.filesys import get_path
from src.designer.use_designer import Designer
W = MarioLevel.default_seg_width

# def collect_batched(path):
#     lvls, simlt_res = [], []
#     path = get_path(path)
#     for batch, name in traverse_batched_level_files(path):
#         lvls += [[lvl[:, s:s+W] for s in range(0, lvl.w, W)] for lvl in batch]
#         with open(f'{path}/{name}_simlt_res.json', 'r') as f:
#             data = json.load(f)
#         simlt_res += [item['seg_infos'] for item in data]
#     return lvls, simlt_res
#     pass
#
# def collect_lvls(path):
#     lvls, simlt_res = [], []
#     path = get_path(path)
#     for lvl, name in traverse_level_files(path):
#         lvls.append([lvl[:, s:s+W] for s in range(0, lvl.w, W)])
#         with open(f'{path}/{name}_simlt_res.json', 'r') as f:
#             data = json.load(f)
#         simlt_res.append(data['seg_infos'])
#     return lvls, simlt_res
#     pass

def make_grid(xticks, y_ticks):
    pass

def get_fun_path(starts, designer, n=4, l=10, stride=1):
    fc_paths, fb_paths = [], []
    start_x, start_y = [], []
    for item in starts:
        obs_buffer = RingQueue(4)
        latvecs = []
        for latvec in item:
            obs_buffer.push(np.array(latvec).astype(np.float32))
            latvecs.append(np.array(latvec).astype(np.float32))

        for i in range(l):
            obs = np.concatenate(obs_buffer.to_list())
            a = designer.act(obs)
            latvecs.append(a)
            obs_buffer.push(a)
        segs = process_levels(generator(torch.tensor(np.array(latvecs)).view(-1, nz, 1, 1)), True)
        full_lvl = repairer.repair(level_sum(segs))
        simlt_res = MarioProxy.get_seg_infos(simulator.simulate_long(full_lvl))
        segs = [full_lvl[:, s:s+W] for s in range(0, full_lvl.w, W)]
        fc_list = fun_c_func.compute_rewards(segs=segs)
        fb_list = fun_b_func.compute_rewards(segs=segs, simlt_res=simlt_res)
        fc_paths.append([np.array(fc_list[i:i+n]).mean() for i in range(n, l, stride)])
        fb_paths.append([np.array(fb_list[i:i+n]).mean() for i in range(n, l, stride)])
        start_x.append(np.array(fc_list[:4]).mean())
        start_y.append(np.array(fb_list[:4]).mean())
    return fc_paths, fb_paths, start_x, start_y

if __name__ == '__main__':
    # designer = Designer('exp_data/main/both/actor.pth')
    generator = get_generator()
    simulator = MarioProxy()
    repairer = DivideConquerRepairer()
    fun_c_func, fun_b_func = FunContent(), FunBehaviour()
    # start_x, start_y = [], []
    # end_x, end_y = [], []
    # n_init = 4
    # l = 5
    # stride = 4
    plt.figure(figsize=(5, 3.85), dpi=320)
    starts = []
    with open(get_path('exp_analysis/extremum_lvls/b_latvec.json'), 'r') as f:
        starts.append(json.load(f))
    with open(get_path('exp_analysis/extremum_lvls/d_latvec.json'), 'r') as f:
        starts.append(json.load(f))
    with open(get_path('exp_analysis/extremum_lvls/g_latvec.json'), 'r') as f:
        starts.append(json.load(f))
    end_x, end_y = [], []
    fc_paths, fb_paths, start_x, start_y = get_fun_path(starts, Designer('exp_data/main/content/actor.pth'))
    fc_paths, fb_paths, *_ = get_fun_path(starts, Designer('exp_data/main/behaviour/actor.pth'))
    fc_paths, fb_paths, *_ = get_fun_path(starts, Designer('exp_data/main/both/actor.pth'))
    fc_paths, fb_paths, *_ = get_fun_path(starts, Designer('exp_data/main/killer/actor.pth'))
    fc_paths, fb_paths, *_ = get_fun_path(starts, Designer('exp_data/main/collector/actor.pth'))
    # for fc_path, fb_path, sx, sy in zip(fc_paths, fb_paths, start_x, start_y):
    #     plt.plot([sx, *fc_path], [sy, *fb_path], lw=1, color='blue', zorder=2)
    #     end_x.append(fc_path[-1])
    #     end_y.append(fb_path[-1])
    # plt.scatter(end_x, end_y, color='blue', marker='*', zorder=3, label='$R_L$', s=40)
    # end_x, end_y = [], []
    #
    # fc_paths, fb_paths, *_ = get_fun_path(starts, Designer('exp_data/main/behaviour/actor.pth'))
    # for fc_path, fb_path, sx, sy in zip(fc_paths, fb_paths, start_x, start_y):
    #     plt.plot([sx, *fc_path], [sy, *fb_path], lw=1, color='green', zorder=2)
    #     end_x.append(fc_path[-1])
    #     end_y.append(fb_path[-1])
    # plt.scatter(end_x, end_y, color='green', marker='*', zorder=3, label='$R_G$', s=40)
    # end_x, end_y = [], []
    #
    # fc_paths, fb_paths, *_ = get_fun_path(starts, Designer('exp_data/main/both/actor.pth'))
    # for fc_path, fb_path, sx, sy in zip(fc_paths, fb_paths, start_x, start_y):
    #     plt.plot([sx, *fc_path], [sy, *fb_path], lw=1, color='red', zorder=2)
    #     end_x.append(fc_path[-1])
    #     end_y.append(fb_path[-1])

    plt.scatter(end_x, end_y, color='red', zorder=3, marker='*', label='$R_G + R_L$', s=40)

    plt.scatter(start_x, start_y, color='black', zorder=3, label='start')
    # plt.scatter(end_x, end_y, color='red', marker='X', s=45, zorder=3)
    ticks = np.linspace(-3, 1, 17)
    plt.xticks(ticks, map(lambda x: '%.2f' % x, ticks), rotation=45.)
    plt.yticks(ticks, map(lambda x: '%.2f' % x, ticks))
    plt.xlim((-2.5, 1.05))
    plt.ylim((-1.5, 1.05))
    plt.xlabel('$f_L$', fontsize=14)
    plt.ylabel('$f_G$ ', fontsize=14, rotation=0.)
    plt.legend()
    # plt.legend(loc='lower right')
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.show()
