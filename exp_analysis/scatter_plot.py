"""
  @Time : 2022/3/19 14:03 
  @Author : Ziqi Wang
  @File : make_map.py 
"""
import json
from itertools import product

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pygame

from smb import traverse_batched_level_files, MarioLevel, traverse_level_files, level_sum
from src.environment.reward_function import FunContent, FunBehaviour
from src.utils.filesys import get_path
from src.utils.mymath import grid_cnt
import config

#
# def collect_orilvls(path, l):
#     W = MarioLevel.default_seg_width
#     ts = MarioLevel.tex_size
#     lvls, simlt_res = [], []
#     for orilvl, name in traverse_level_files(path):
#         with open(get_path(f'{path}/{name}_simlt_res.json'), 'r') as f:
#             full_trace = json.load(f)['full_trace']
#
#         for s in range(0, orilvl.w - W * l):
#             segs = [orilvl[:, s+i*W:s+(i+1)*W] for i in range(l)]
#             lvls.append(segs)
#
#             simlt_res.append([])
#             i, p = 0, 0
#             while full_trace[p][0] < s * ts:
#                 p += 1
#             while i < l:
#                 p0 = p
#                 while p < len(full_trace) and full_trace[p][0] < ((i + 1) * W + s) * ts:
#                     p += 1
#                 simlt_res[-1].append(
#                     {'trace': [[x-full_trace[p0][0], y] for x, y in full_trace[p0:p]]}
#                 )
#                 i += 1
#     return lvls, simlt_res
#
# def collect_lvls(path):
#     W = MarioLevel.default_seg_width
#     lvls, simlt_res = [], []
#     for batch, name in traverse_batched_level_files(path):
#         lvls += [[lvl[:, s:s+W] for s in range(0, lvl.w, W)] for lvl in batch]
#         with open(get_path(f'{path}/{name}_simlt_res.json'), 'r') as f:
#             data = json.load(f)
#         simlt_res += [item['seg_infos'] for item in data]
#     return lvls, simlt_res
#     pass
#
# def make(gc, gb, lvls, simlt_res, folder='orilvl'):
#     # path = f'exp_data/randlvls/l{l}'
#
#
#     scatter = True
#     heat = False
#
#     fun_c_func = FunContent(g=gc)
#     fun_b_func = FunBehaviour(g=gb)
#
#
#     fun_c = [
#         fun_c_func.compute_rewards(segs=segs)[-1]
#         for segs in lvls
#     ]
#     print(fun_c)
#     fun_b = [
#         fun_b_func.compute_rewards(segs=segs, simlt_res=item)[-1]
#         for segs, item in zip(lvls, simlt_res)
#     ]
#     print(fun_b)
#
#     if scatter:
#         plt.figure(figsize=(4.2, 4.2), dpi=200)
#         plt.scatter(fun_c, fun_b, alpha=0.2, color='blue', linewidths=0)
#         plt.title(f'gc={gc}, gb={gb}')
#         plt.xlabel('fun-content')
#         plt.ylabel('fun-behaviour')
#         plt.xlim((-0.5, 1.05))
#         plt.ylim((-0.5, 1.05))
#         plt.grid()
#         plt.tight_layout()
#         # print(f'./figs/{folder}/gc{gc:.3f}_gb{gb:.3f}.png')
#         # plt.show()
#         plt.savefig(get_path(f'exp_analyze/figs/{folder}/gc{gc:.3f}_gb{gb:.3f}.png'))
#
# def viz(lvl, trace, path, name):
#     lvl.to_img(f'{path}/{name}.png')
#     img_with_trace = lvl.to_img(None)
#     pygame.draw.lines(img_with_trace, 'black', False, [(x, y - 8) for x, y in trace], 3)
#     pygame.image.save(img_with_trace, f'{path}/{name}_with_trace.png')


gc_lb, gc_ub = 0.075, 0.125
gb_lb, gb_ub = 0.15, 0.35
n_rows, n_cols = int((gb_ub - gb_lb + 1e-4) / 0.05) + 1, int((gc_ub - gc_lb + 1e-4) / 0.025) + 1


def compute_index(gc, gb):
    return int(round((gb - gb_lb) / 0.05)), int(round((gc - gc_lb) / 0.025))
    pass


if __name__ == '__main__':
    # X axis: behavior, Y axis: content
    # rows: gb, cols: gc
    fig_size=3
    dpi=200
    bounds = {}
    plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    for n in [3 ,4]:
        with open(get_path(f'exp_analysis/fun_statistics/n{n}.json'), 'r') as f:
            data = json.load(f)
        for item in data:
            gc_val, gb_val = item['gc'], item['gb']
            i, j = compute_index(gc_val, gb_val)
            # print(i, j, n_rows, n_cols)
            if 0 <= i < n_rows and 0 <= j < n_cols:
                x1, y1 = item['rand_fun_c'], item['rand_fun_b']
                x2, y2 = item['ori_fun_c'], item['ori_fun_b']
                plt.scatter(x1, y1, color='red', alpha=0.12, linewidths=0, s=15)
                plt.scatter(x2, y2, color='blue', alpha=0.12, linewidths=0, s=15)
                plt.xlabel('$f_c$', size=12)
                plt.ylabel('$f_b$', size=12, rotation=0.)
                plt.title(f'gc={gc_val:.3f}, gb={gb_val:.3f}')
                plt.grid()
                # if n_init <= 4:
                plt.xlim((-4.5, 1.15))
                plt.ylim((-4.5, 1.15))
                # else:
                # plt.xlim((-3, 1.1))
                # plt.ylim((-3, 1.1))
                ticks = [-4., -3., -2., -1., 0., 1.]
                plt.xticks(ticks, map(lambda v: '%.1f' % v, ticks))
                plt.yticks(ticks, map(lambda v: '%.1f' % v, ticks))
                plt.tight_layout()
                plt.savefig(get_path(f'exp_analysis/fun_scatter_plots/n{n}/{i}_{j}.png'))
                plt.cla()
        size = int(fig_size * dpi)
        full_img = PIL.Image.new('RGB', (size * n_rows, size * n_cols))
        print(n_cols, n_rows)
        for i, j in product(range(n_rows), range(n_cols)):
            img = PIL.Image.open(get_path(f'exp_analysis/fun_scatter_plots/n{n}/{i}_{j}.png'))
            full_img.paste(img, (size * i, size * j, size * (i + 1), size * (j + 1)))
            full_img.save(get_path(f'exp_analysis/fun_scatter_plots/scatter_sheets{n}.png'))
            pass


    # with open(get_path(f'exp_analysis/fun_statistics/n4.json'), 'r') as f:
    #     data = json.load(f)
    # plt.figure(figsize=(5, 3.85), dpi=320)
    # extrem_x = [
    #     0.9606093285828107,
    #     0.850261829815147,
    #     0.8191471952180664,
    #     -2.270069851922988,
    #     0.6050662342265672,
    #     -0.761134909810617,
    #     0.6106867140677432,
    #     0.8938169699536926,
    #     # -0.3596720176029186
    # ]
    # extrem_y = [
    #     1.0,
    #     -0.7492504392749701,
    #     0.5113329382532883,
    #     0.8872145848368109,
    #     0.8015010020042267,
    #     -0.5035509241664601,
    #     0.6386411384953374,
    #     0.9476561646449732,
    #     # 0.6785450334197067
    # ]
    # labels = ['ztraces', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    # for x, y, l in zip(extrem_x, extrem_y, labels):
    #     plt.text(x+0.05, y-0.09, l, color='black', ha='center', va='center', size=12)
    #
    # for item in data:
    #     gc_val, gb_val = item['gc'], item['gb']
    #     # i, j = compute_index(gc_val, gb_val)
    #     # print(i, j, n_rows, n_cols)
    #     if abs(gc_val - 0.1) < 1e-4 and abs(gb_val - 0.25) < 1e-4:
    #         x1, y1 = item['rand_fun_c'], item['rand_fun_b']
    #         x2, y2 = item['ori_fun_c'], item['ori_fun_b']
    #         # x3, y3 = both_data['fun_c'], both_data['fun_b']
    #         # x4, y4 = content_data['fun_c'], content_data['fun_b']
    #         # x5, y5 = behaviour_data['fun_c'], behaviour_data['fun_b']
    #         # print(x3, y3, sep='\n_init')
    #         plt.scatter(x1, y1, color='red', alpha=0.16, linewidths=0, s=16)
    #         plt.scatter(x2, y2, color='blue', alpha=0.16, linewidths=0, s=16)
    #
    #         plt.scatter([-10], [-10], color='red', linewidths=0, s=16, label='randomly generated levels')
    #         plt.scatter([-10], [-10], color='blue', linewidths=0, s=16, label='human-designed levels')
    #         plt.scatter(extrem_x, extrem_y, color='black', linewidths=0, marker='*', s=56, label='selected levels')
    #         # plt.scatter(x3, y3, color=(0.25, 0.25, 0.25), alpha=0.5, linewidths=0, marker='*', s=28)
    #         # plt.scatter(x4, y4, color=(0.25, 0.25, 0.25), alpha=0.5, linewidths=0, marker='*', s=28)
    #         # plt.scatter(x5, y5, color=(0.25, 0.25, 0.25), alpha=0.5, linewidths=0, marker='*', s=28)
    #         plt.xlabel('$f_L$', size=13)
    #         plt.ylabel('$f_G$', size=13, rotation=0.)
    #         xticks, yticks = np.linspace(-3, 1, 17), np.linspace(-3, 1, 17)
    #         plt.xticks(xticks, map(lambda x: '%.2f' % x, xticks), size=8, rotation=45.)
    #         plt.yticks(yticks, map(lambda x: '%.2f' % x, yticks), size=8)
    #         # plt.title(f'gc={gc_val:.3f}, gb={gb_val:.3f}')
    #         plt.grid()
    #         plt.xlim((-2.5, 1.05))
    #         plt.ylim((-1.5, 1.05))
    #         plt.legend(fontsize=10)
    #         plt.tight_layout()
    #         plt.savefig(get_path(f'exp_analysis/fun_scatter_plots/scatter_used.png'))
    #
