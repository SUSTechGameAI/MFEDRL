# """
#   @Time : 2022/1/1 16:55
#   @Author : Ziqi Wang
#   @File : level_divs.py
# """
#

import numpy as np
from dtw import dtw
from smb import MarioLevel
from src.utils.mymath import jsdiv


def trace_div(trace1, trace2, w=10):
    ts = MarioLevel.tex_size
    t1, t2 = np.array(trace1) / ts, np.array(trace2) / ts
    dist_metric = (lambda x, y: np.linalg.norm(x - y))
    dtw_val, *_ = dtw(t1, t2, dist_metric, w=max(w, abs(len(t1) - len(t2))))
    return dtw_val / MarioLevel.default_seg_width

def tile_pattern_js_div(seg1: MarioLevel, seg2: MarioLevel, w=2):
    counts1 = seg1.tile_pattern_counts(w)
    counts2 = seg2.tile_pattern_counts(w)
    all_keys = counts1.keys().__or__(counts2.keys())
    p = np.array([counts1.setdefault(key, 0) for key in all_keys])
    q = np.array([counts2.setdefault(key, 0) for key in all_keys])
    return jsdiv(p, q)

