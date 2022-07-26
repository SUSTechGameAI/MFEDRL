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
    # return dtw_val / (MarioLevel.tex_size * MarioLevel.default_seg_width)
    return dtw_val / max(len(t1), len(t2))

def tile_pattern_js_div(seg1: MarioLevel, seg2: MarioLevel, w=2):
    counts1 = seg1.tile_pattern_counts(w)
    counts2 = seg2.tile_pattern_counts(w)
    all_keys = counts1.keys().__or__(counts2.keys())
    p = np.array([counts1.setdefault(key, 0) for key in all_keys])
    q = np.array([counts2.setdefault(key, 0) for key in all_keys])
    return jsdiv(p, q)

#
# def tile_diff_rate(seg1: MarioLevel, seg2: MarioLevel):
#     h, w = seg1.shape
#     diff = np.where(seg1.content != seg2.content, 1, 0).sum()
#     return diff / (h * w)
#
# def tile_pattern_kl_div(seg1: MarioLevel, seg2: MarioLevel, w):
#     eps = 1e-3
#     counts1 = seg1.tile_pattern_counts(w)
#     counts2 = seg2.tile_pattern_counts(w)
#     all_keys = counts1.keys().__or__(counts2.keys())
#     revised_counts1 = np.array([counts1.setdefault(key, 0) + eps for key in all_keys])
#     revised_counts2 = np.array([counts2.setdefault(key, 0) + eps for key in all_keys])
#     return entropy(revised_counts1, revised_counts2, base=2)
#
#
#
# class ContentDivergenceMetrics(Enum):
#     TileDiffRate = 0
#     TilePttrJS2 = 1
#     TilePttrJS3 = 2
#     TilePttrJS4 = 3
#     TilePttrJS23 = 4
#     TilePttrKL2 = 5
#
#     def get_func(self):
#         if self.name == 'TileDiffRate':
#             return tile_diff_rate
#         elif self.name == 'TilePttrJS2':
#             return lambda a, b: tile_pattern_js_div(a, b, 2)
#         elif self.name == 'TilePttrJS3':
#             return lambda a, b: tile_pattern_js_div(a, b, 3)
#         elif self.name == 'TilePttrJS4':
#             return lambda a, b: tile_pattern_js_div(a, b, 4)
#         elif self.name == 'TilePttrJS23':
#             return lambda a, b: (
#                 tile_pattern_js_div(a, b, 2) +
#                 tile_pattern_js_div(a, b, 3)
#             ) / 2
#         elif self.name == 'TilePttrKL2':
#             return lambda a, b: tile_pattern_kl_div(a, b, 2)
#
#     def __str__(self):
#         return self.name
#
