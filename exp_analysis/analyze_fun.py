"""
  @Time : 2022/3/24 16:09 
  @Author : Ziqi Wang
  @File : analyze_fun.py 
"""
import sys
sys.path.append('../')

import json
import time
import numpy as np
from itertools import product
from src.utils.filesys import get_path
from src.utils.parallel import MyAsyncPool
from smb import traverse_batched_level_files, MarioLevel, traverse_level_files
from src.environment.reward_function import FunContent, FunBehaviour


def collect_orilvls(path, l):
    W = MarioLevel.default_seg_width
    ts = MarioLevel.tex_size
    lvls, simlt_res = [], []
    for orilvl, name in traverse_level_files(path):
        with open(get_path(f'{path}/{name}_simlt_res.json'), 'r') as f:
            full_trace = json.load(f)['full_trace']

        for s in range(0, orilvl.w - W * l):
            segs = [orilvl[:, s+i*W:s+(i+1)*W] for i in range(l)]
            lvls.append(segs)

            simlt_res.append([])
            i, p = 0, 0
            while full_trace[p][0] < s * ts:
                p += 1
            while i < l:
                p0 = p
                while p < len(full_trace) and full_trace[p][0] < ((i + 1) * W + s) * ts:
                    p += 1
                simlt_res[-1].append(
                    {'trace': [[x-full_trace[p0][0], y] for x, y in full_trace[p0:p]]}
                )
                i += 1
    return lvls, simlt_res

def collect_batched_lvls(path):
    W = MarioLevel.default_seg_width
    lvls, simlt_res = [], []
    for batch, name in traverse_batched_level_files(path):
        lvls += [[lvl[:, s: s + W] for s in range(0, W*5, W)] for lvl in batch]
        with open(get_path(f'{path}/{name}_simlt_res.json'), 'r') as f:
            data = json.load(f)
        simlt_res += [item['seg_infos'][:5] for item in data]
    return lvls, simlt_res

def compute_fun_vals(gc, gb, n):
    ori_lvls, ori_simlt = collect_orilvls('exp_data/orilvls', 5)
    rand_lvls, rand_simlt = collect_batched_lvls('exp_data/rand_playable_lvls2')
    print(len(ori_lvls), len(rand_lvls))
    fun_c_func = FunContent(g=gc, n=n)
    fun_b_func = FunBehaviour(g=gb, n=n)
    ori_fun_c = [
        sum(fun_c_func.compute_rewards(segs=segs)) / 4
        for segs in ori_lvls
    ]
    ori_fun_b = [
        sum(fun_b_func.compute_rewards(segs=segs, simlt_res=item)) / 4
        for segs, item in zip(ori_lvls, ori_simlt)
    ]
    rand_fun_c = [
        sum(fun_c_func.compute_rewards(segs=segs)) / 4
        for segs in rand_lvls
    ]
    rand_fun_b = [
        sum(fun_b_func.compute_rewards(segs=segs, simlt_res=item)) / 4
        for segs, item in zip(rand_lvls, rand_simlt)
    ]
    return {
        'gc': gc, 'gb': gb,
        'ori_fun_c': ori_fun_c,
        'ori_fun_b': ori_fun_b,
        'rand_fun_c': rand_fun_c,
        'rand_fun_b': rand_fun_b
    }

if __name__ == '__main__':
    # lvls, simlt = collect_batched_lvls('exp_data/main/behaviour/for_scatter')
    # print(len(lvls), len(lvls))
    # fun_c_func = FunContent()
    # fun_b_func = FunBehaviour()
    # fun_c = [
    #     np.array(fun_c_func.compute_rewards(segs=segs)[4:]).mean()
    #     for segs in lvls
    # ]
    # fun_b = [
    #     np.array(fun_b_func.compute_rewards(segs=segs, simlt_res=item)[4:]).mean()
    #     for segs, item in zip(lvls, simlt)
    # ]
    # with open(get_path('exp_data/main/behaviour/for_scatter/fun_vals.json'), 'w') as f:
    #     json.dump({'fun_c': fun_c, 'fun_b': fun_b}, f)
    parallel = 5
    gcs = np.linspace(0.075, 0.15, 4)
    gbs = np.linspace(0.15, 0.35, 5)
    computing_pool = MyAsyncPool(parallel)
    # gcs = [0.2]
    # gbs = [0.35]
    ns = [3, 4, 5]
    for nval in ns:
        res = []
        start_time = time.time()
        for gcval, gbval in product(gcs, gbs):
            computing_pool.push(compute_fun_vals, (gcval, gbval, nval))
            while computing_pool.get_num_waiting() > 2 * parallel:
                res += computing_pool.collect()
                time.sleep(1)
        res += computing_pool.wait_and_get()
        with open(get_path(f'exp_analysis/new_normalise/fun_statistics/n{nval}.json'), 'w') as f:
            json.dump(res, f)

