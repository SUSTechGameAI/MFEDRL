"""
  @Time : 2022/2/6 13:44 
  @Author : Ziqi Wang
  @File : orilvl_statistics.py
"""

import json
from itertools import product

import matplotlib.pyplot as plt
from math import ceil

import numpy as np

from src.utils.filesys import get_path
from smb import MarioLevel, MarioProxy, traverse_level_files
from src.level_divs import trace_div, tile_pattern_js_div


def test_traces(agent=None, save_path='./', save_fname='traces.json'):
    results = {}
    proxy = MarioProxy()
    for lvl, name in traverse_level_files('levels/train'):
        kwargs = {} if agent is None else {'agent': agent}
        results[name] = proxy.simulate_long(lvl, **kwargs)['trace']

    with open(get_path(save_path + f'/{save_fname}'), 'w') as f:
        json.dump(results, f)
    print('Trace test finish')

def test_content_divs(metric, delta=1, save_path='./', save_name='content_div.json'):
    W = MarioLevel.default_seg_width
    results = {}
    for lvl, name in traverse_level_files('levels/train'):
        results[name] = []
        for s in range(W * delta, lvl.w - W):
            seg1 = lvl[:, s - delta * W: s - (delta - 1) * W]
            seg2 = lvl[:, s: s + W]
            results[name].append(metric(seg1, seg2))

    with open(get_path(save_path + f'/{save_name}'), 'w') as f:
        json.dump(results, f)
    print('Content divergence test finish')
    pass

def test_behaviour_divs(trace_files, delta=1, save_path='./', save_name=None):
    W = MarioLevel.default_seg_width
    ts = MarioLevel.tex_size
    results = {}
    with open(get_path(trace_files), 'r') as f:
        data = json.load(f)
    for name, trace in data.items():
        results[name] = []
        end = ceil(trace[-1][0] / ts) + 1
        trace.append([float('inf'), 0])
        # get traces slices for each window
        trace_slices = []
        i, j = 0, 0
        for s in range(0, end - W):
            e = s + W
            while trace[i][0] < s * ts:
                i += 1
            while trace[j][0] < e * ts:
                j += 1
            trace_slices.append([[item[0] - s * ts, item[1]] for item in trace[i: j-1]])
        # compute behavior divergences
        for i in range(0, len(trace_slices) - W * delta):
            j = i + W * delta
            results[name].append(trace_div(trace_slices[i], trace_slices[j]))
    if save_name is None:
        save_name = 'behaviour_divs.json'
    with open(get_path(save_path + f'/{save_name}'), 'w') as f:
        json.dump(results, f)
    print('Behaviour divergence test finish')
    pass

def test_metric_correlation(files1, files2):
    res = []
    for file1, file2 in zip(files1, files2):
        with open(get_path(file1), 'r') as f1, open(get_path(file2), 'r') as f2:
            divs1, divs2 = [], []
            for item1, item2 in zip(json.load(f1).values(), json.load(f2).values()):
                divs1 += item1
                divs2 += item2
        x, y = np.array(divs1), np.array(divs2)
        res.append(np.corrcoef(x, y)[0, 1])
    return res

def viz_divs(names, deltas, data, minmax=(None, None), value_text=True, title=''):
    fig = plt.figure(figsize=(4.8, 2.4), dpi=300)
    ax = fig.subplots()
    vmin, vmax = minmax
    print(data)
    ax.imshow(data, vmin=vmin, vmax=vmax)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(names)), labels=names, size=8)
    ax.set_yticks(np.arange(len(deltas)), labels=deltas, size=8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if value_text:
        for i, j in product(range(len(deltas)), range(len(names))):
            ax.text(j, i, '%.2g' % data[i][j], ha="center", va="center", color='white', size=7)

    ax.set_title(title, fontdict={'fontsize': 10})
    fig.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    plt.figure(dpi=200)
    names = [name for _, name in traverse_level_files()]
    for delta in range(1, 5):
        c_divs = []
        b_divs = []
        with open(get_path(f'exp_data/orilvl_statistics/intermediate/content_JS2_delta{delta}.json'), 'r') as f:
            raw_data = json.load(f)
        for name in names:
            div_vals = np.array(raw_data[name])
            c_divs.append(div_vals.mean())
        with open(get_path(f'exp_data/orilvl_statistics/intermediate/runner_div_delta{delta}.json'), 'r') as f:
            raw_data = json.load(f)
        for name in names:
            div_vals = np.array(raw_data[name])
            b_divs.append(div_vals.mean())
        plt.scatter(c_divs, b_divs, label=f'shifts={delta}')
    plt.xlabel('Content dissimilarity')
    plt.ylabel('Behaviour dissimilarity')
    plt.legend(loc='upper right')
    plt.show()
    pass
