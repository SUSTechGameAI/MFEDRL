"""
  @Time : 2022/3/16 11:15 
  @Author : Ziqi Wang
  @File : numerical_eval.py 
"""
import sys
sys.path.append('../')

import json
import argparse
import importlib
from config import history_len
from smb import traverse_level_files, MarioLevel
from src.utils.filesys import get_path
from src.level_divs import tile_pattern_js_div, trace_div


def compute_novelty(hist_len, metric, content):
    novelty_sum, novelty_cnt = [0.] * hist_len, [0] * hist_len
    for i in range(1, len(content)):
        for k in range(1, hist_len + 1):
            j = i - k
            if j < 0:
                break
            novelty_sum[k-1] = novelty_sum[k-1] + metric(content[i], content[j])
            novelty_cnt[k-1] = novelty_cnt[k-1] + 1
    return [novelty_sum[i] / novelty_cnt[i] for i in range(hist_len)]
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--rfunc', type=str, default='')
    parser.add_argument('--c_novelty', type=int, default=history_len)
    parser.add_argument('--b_novelty', type=int, default=history_len)

    args = parser.parse_args()

    rfunc = (
        importlib.import_module('src.environment.rfuncs')
        .__getattribute__(f'{args.rfunc}')
    ) if args.rfunc != '' else None
    res = []
    for lvl, name in traverse_level_files(args.path):
        trial_res = {}
        w = MarioLevel.default_seg_width
        segs = [lvl[:, s:s+w] for s in range(0, lvl.w, w)]
        if (rfunc is not None and rfunc.require_simlt) or args.b_novelty > 0:
            with open(get_path(f'{args.path}/{name}_simlt_res.json'), 'r') as f:
                simlt_res = json.load(f)['seg_infos']
        else:
            simlt_res = None
        if rfunc is not None:
            rewards = rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            trial_res = {key: sum(val) / (len(val) - 1) for key, val in rewards.items()}
        if args.c_novelty > 0:
            trial_res['content_novelties'] = compute_novelty(args.c_novelty, tile_pattern_js_div, segs)
        if args.b_novelty > 0:
            seg_traces = [item['trace'] for item in simlt_res]
            trial_res['behavior_novelties'] = compute_novelty(args.c_novelty, trace_div, seg_traces)
        res.append(trial_res)
    with open(get_path(f'{args.path}/evaluations.json'), 'w') as f:
        json.dump(res, f)
