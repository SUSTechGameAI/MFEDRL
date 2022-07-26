"""
  @Time : 2022/3/27 22:08 
  @Author : Ziqi Wang
  @File : viz_starts.py 
"""


import json

import matplotlib.pyplot as plt

from smb import MarioLevel, traverse_batched_level_files, traverse_level_files
from src.environment.reward_function import FunContent, FunBehaviour
from src.utils.filesys import get_path

if __name__ == '__main__':
    with open(get_path('exp_data/starts100.json'), 'r') as f:
        data = json.load(f)
    x = [item['fc'] for item in data]
    y = [item['fb'] for item in data]
    plt.figure(figsize=(4.8, 4.8), dpi=200)
    plt.scatter(x, y, color='red')
    # plt.xlim((-2, 1.05))
    # plt.ylim((-2, 1.05))
    plt.xlabel('fun-content')
    plt.ylabel('fun-behaviour')
    plt.grid()
    plt.tight_layout()
    plt.show()
        # print(fun_b)
    #     if all(item['playable'] for item in simlt):
    #         used_trials.append({})
    #
    #     if len(used_trials) == 30:
    #         break
    # pass
