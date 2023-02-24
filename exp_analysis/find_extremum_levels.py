"""
  @Time : 2022/3/22 14:08 
  @Author : Ziqi Wang
  @File : findlevel.py 
"""
import json

from smb import MarioLevel, traverse_batched_level_files, level_sum
from src.environment.reward_function import FunContent, FunBehaviour
from src.utils.filesys import get_path


def collect_lvls(path):
    W = MarioLevel.default_seg_width
    lvls, simlt_res, latvecs = [], [], []
    for batch, name in traverse_batched_level_files(path):
        lvls += [[lvl[:, i * W: (i+1) * W] for i in range(8)] for lvl in batch]
        with open(get_path(f'{path}/{name}_simlt_res.json'), 'r') as f:
            data = json.load(f)
        simlt_res += data
        with open(get_path(f'{path}/{name}_latvecs.json'), 'r') as f:
            data = json.load(f)
        latvecs += data
        # simlt_res += [item['seg_infos'][:l] for item in data]
    return lvls, simlt_res, latvecs

if __name__ == '__main__':
    gc, gb, n = 0.1, 0.25, 4
    # l = n_init+1
    lvls, simlt_res, latvecs = collect_lvls('exp_data/rand_playable_lvls2')
    full_traces = [item['full_trace'] for item in simlt_res]

    fun_c_func = FunContent(g=gc, n=n)
    fun_b_func = FunBehaviour(g=gb, n=n)
    fun_c_plus_func = FunContent(g=gc, n=n, mode=1)
    fun_b_plus_func = FunBehaviour(g=gb, n=n, mode=1)
    fun_c_minus_func = FunContent(g=gc, n=n, mode=-1)
    fun_b_minus_func = FunBehaviour(g=gb, n=n, mode=-1)

    fun_c_vals = [
        sum(fun_c_func.compute_rewards(segs=segs[:5])) / 4
        for segs in lvls
    ]
    fun_b_vals = [
        sum(fun_b_func.compute_rewards(segs=segs[:5], simlt_res=item[:5])) / 4
        for segs, item in zip(lvls, [item['seg_infos'] for item in simlt_res])
    ]
    fun_c_plus_vals = [
        sum(fun_c_plus_func.compute_rewards(segs=segs[:5])) / 4
        for segs in lvls
    ]
    fun_b_plus_vals = [
        sum(fun_b_plus_func.compute_rewards(segs=segs[:5], simlt_res=item[:5])) / 4
        for segs, item in zip(lvls, [item['seg_infos'] for item in simlt_res])
    ]
    fun_c_minus_vals = [
        sum(fun_c_minus_func.compute_rewards(segs=segs[:5])) / 4
        for segs in lvls
    ]
    fun_b_minus_vals = [
        sum(fun_b_minus_func.compute_rewards(segs=segs[:5], simlt_res=item[:5])) / 4
        for segs, item in zip(lvls, [item['seg_infos'] for item in simlt_res])
    ]

    i1 = max(
        *range(1000),
        key=lambda i: fun_c_vals[i] + fun_b_vals[i]
    )
    name1 = 'pcpb'
    print(full_traces[i1])
    level_sum(lvls[i1][:5]).to_img_with_trace(full_traces[i1], f'exp_analysis/extremum_lvls/{name1}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name1}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i1], 'fb': fun_b_vals[i1]}, f)

    i2 = min(
        *(i for i in range(1000) if fun_c_vals[i] > 0.75),
        key=lambda i: fun_b_plus_vals[i]
    )
    name2 = 'pclb'
    level_sum(lvls[i2][:5]).to_img_with_trace(full_traces[i2], f'exp_analysis/extremum_lvls/{name2}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name2}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i2], 'fb': fun_b_vals[i2]}, f)

    i3 = min(
        *(i for i in range(1000) if fun_c_vals[i] > 0.75),
        key=lambda i: fun_b_minus_vals[i]
    )
    name3 = 'pchb'
    level_sum(lvls[i3][:5]).to_img_with_trace(full_traces[i3], f'exp_analysis/extremum_lvls/{name3}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name3}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i3], 'fb': fun_b_vals[i3]}, f)
    with open(get_path(f'exp_analysis/extremum_lvls/b_latvec.json'), 'w') as f:
        json.dump(latvecs[i3][:5], f)

    i4 = min(
        *(i for i in range(1000) if fun_b_vals[i] > 0.75),
        key=lambda i: fun_c_plus_vals[i]
    )
    name4 = 'lcpb'
    level_sum(lvls[i4][:5]).to_img_with_trace(full_traces[i4], f'exp_analysis/extremum_lvls/{name4}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name4}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i4], 'fb': fun_b_vals[i4]}, f)

    i5 = min(
        *(i for i in range(1000) if fun_b_vals[i] > 0.75),
        key=lambda i: fun_c_minus_vals[i]
    )
    name5 = 'hcpb'
    level_sum(lvls[i5][:5]).to_img_with_trace(full_traces[i5], f'exp_analysis/extremum_lvls/{name5}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name5}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i5], 'fb': fun_b_vals[i5]}, f)
    with open(get_path(f'exp_analysis/extremum_lvls/d_latvec.json'), 'w') as f:
        json.dump(latvecs[i5][:5], f)

    i6 = min(
        *range(1000),
        key=lambda i: fun_b_plus_vals[i] + fun_c_plus_vals[i]
    )
    name6 = 'lclb'
    level_sum(lvls[i6][:5]).to_img_with_trace(full_traces[i6], f'exp_analysis/extremum_lvls/{name6}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name6}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i6], 'fb': fun_b_vals[i6]}, f)

    i7 = min(
        *(i for i in range(1000) if 0.8 < (1 - fun_c_vals[i] + 1e-5) / (1 - fun_b_vals[i] + 1e-5) < 1.25),
        key=lambda i: fun_b_minus_vals[i] + fun_c_minus_vals[i]
    )
    name7 = 'hchb'
    level_sum(lvls[i7][:5]).to_img_with_trace(full_traces[i7], f'exp_analysis/extremum_lvls/{name7}.png')
    with open(get_path(f'exp_analysis/extremum_lvls/{name7}.json'), 'w') as f:
        json.dump({'fc': fun_c_vals[i7], 'fb': fun_b_vals[i7]}, f)
    with open(get_path(f'exp_analysis/extremum_lvls/g_latvec.json'), 'w') as f:
        json.dump(latvecs[i7][:5], f)
    # pass
