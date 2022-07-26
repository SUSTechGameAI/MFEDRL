"""
  @Time : 2022/4/5 19:28 
  @Author : Ziqi Wang
  @File : analyze_vary_players.py 
"""
import json

import numpy as np

from smb import traverse_level_files, MarioLevel
from src.environment.reward_function import FunContent, FunBehaviour, Playability
from src.utils.filesys import get_path

if __name__ == '__main__':
    designer = 'collector'
    fc_func = FunContent()
    fb_func = FunBehaviour()
    p_func = Playability()
    W = MarioLevel.default_seg_width
    folder = f'exp_data/main/{designer}/vary_player'
    results = {
        key: [] for key in (
            'fb-r', 'fb-k', 'fb-c',
            'p-r', 'p-k', 'p-c',
            'fc'
        )
    }
    for lvl, name in traverse_level_files(folder):
        segs = [lvl[:, s:s + W] for s in range(0, lvl.w, W)]
        with open(get_path(f'{folder}/{name}_runner_simlt_res.json'), 'r') as f:
            simlt_res_r = json.load(f)['seg_infos']
        with open(get_path(f'{folder}/{name}_killer_simlt_res.json'), 'r') as f:
            simlt_res_k = json.load(f)['seg_infos']
        with open(get_path(f'{folder}/{name}_collector_simlt_res.json'), 'r') as f:
            simlt_res_c = json.load(f)['seg_infos']
        results['fb-r'].append(np.array(fb_func.compute_rewards(segs=segs, simlt_res=simlt_res_r)[3:]).mean())
        results['p-r'].append(np.array(p_func.compute_rewards(segs=segs, simlt_res=simlt_res_r)[3:]).mean())
        results['fb-k'].append(np.array(fb_func.compute_rewards(segs=segs, simlt_res=simlt_res_k)[3:]).mean())
        results['p-k'].append(np.array(p_func.compute_rewards(segs=segs, simlt_res=simlt_res_k)[3:]).mean())
        results['fb-c'].append(np.array(fb_func.compute_rewards(segs=segs, simlt_res=simlt_res_c)[3:]).mean())
        results['p-c'].append(np.array(p_func.compute_rewards(segs=segs, simlt_res=simlt_res_c)[3:]).mean())
        results['fc'].append(np.array(fc_func.compute_rewards(segs=segs)[3:]).mean())

    with open(get_path(f'{folder}/test_rewards.json'), 'w') as f:
        json.dump(results, f)
