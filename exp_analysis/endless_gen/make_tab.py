"""
  @Time : 2022/10/14 14:35 
  @Author : Ziqi Wang
  @File : make_tab.py 
"""

import json
import numpy as np
import pandas as pds
from itertools import product
from src.utils.filesys import get_path


if __name__ == '__main__':
    divstab = pds.read_csv(get_path('exp_analysis/endless_gen/results/lvsdivs-dtw.csv'), dtype=str)
    randdivs = pds.read_csv(get_path('exp_analysis/endless_gen/results/randdivs.csv'), index_col='n')
    # with open(get_path('exp_analysis/endless_gen/rand_dist.json'), 'r') as f:
    # print(divstab)
    # print(divstab[(divstab['gamma'] == '080') & (divstab['n'] == '4')]['diversity'].values[0])
    rls1 = []
    rls2 = []
    rls3 = []
    # rgs = []
    # rps = []
    dvs = []
    for n, gm in product((4, 6), ('070', '080', '090', '099')):
        rewards = np.load(get_path(f'exp_analysis/endless_gen/data2/gm{gm}n{n}_rewards.npy'))
        print(rewards.shape)
        rls1.append(rewards[:, -50:-40].mean())
        rls2.append(rewards[:, -40:-25].mean())
        rls3.append(rewards[:, -25:].mean())
        d0 = randdivs.loc[n]['diversity']
        tmp = divstab[(divstab['gamma'] == gm) & (divstab['n'] == str(n))]['diversity']
        # print(tmp)
        divs = float(tmp.values[0]) / d0
        dvs.append(divs)
    print(' & '.join([r'$\bar R_{1:10}$', *['%.3f' % v for v in rls1]]) + r' \\')
    print(' & '.join([r'$\bar R_{11:25}$', *['%.3f' % v for v in rls2]]) + r' \\')
    print(' & '.join([r'$\bar R_{26:50}$', *['%.3f' % v for v in rls3]]) + r' \\')
    print('\\hline')
    print(' & '.join(['Div', *['%.3f' % v for v in dvs]]) + r' \\')
    pass
