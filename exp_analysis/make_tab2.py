"""
  @Time : 2022/4/5 19:55 
  @Author : Ziqi Wang
  @File : make_tab2.py 
"""
import json
import numpy as np
from src.utils.filesys import get_path

if __name__ == '__main__':
    tab_lines = {
        '\\textit{runner}': 'both',
        '\\textit{killer}': 'killer',
        '\\textit{collector}': 'collector',
    }
    for txt, fd in tab_lines.items():
        with open(get_path(f'exp_data/main/{fd}/vary_player/test_rewards.json'), 'r') as f:
            data = json.load(f)
        rewards = {key: np.array(val) for key, val in data.items()}

        fmt =  txt + (' & %.3f $\pm$ %.2f' * 7) + ' \\\\'
        print(
            fmt % (
                rewards['fb-r'].mean(), rewards['fb-r'].std(),
                rewards['fb-k'].mean(), rewards['fb-k'].std(),
                rewards['fb-c'].mean(), rewards['fb-c'].std(),
                rewards['p-r'].mean(), rewards['p-r'].std(),
                rewards['p-k'].mean(), rewards['p-k'].std(),
                rewards['p-c'].mean(), rewards['p-c'].std(),
                rewards['fc'].mean(), rewards['fc'].std(),
            )
        )
