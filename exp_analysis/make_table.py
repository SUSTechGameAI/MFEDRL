"""
  @Time : 2022/3/29 11:12 
  @Author : Ziqi Wang
  @File : make_table.py 
"""

import json
import numpy as np
from src.utils.filesys import get_path


if __name__ == '__main__':
    keys = ['FunContent', 'FunBehaviour', 'Playability']
    for folder in ['both', 'content', 'behaviour']:
        with open(get_path(f'exp_data/main/{folder}/test_rewards.json'), 'r') as f:
            data = json.load(f)
        # for key in ['FunContent', 'FunBehaviour', 'Playability']:
        data = {key: np.array(val)/25 for key, val in data.items()}
        print(' & '.join('%.3f $\pm$ %.2f' % (data[key].mean(), data[key].std()) for key in keys), '\\\\')

    with open(get_path(f'exp_data/randgen/test_rewards.json'), 'r') as f:
        data = json.load(f)
    data = {key: np.array(val)/25 for key, val in data.items()}
    print(' & '.join('%.3f $\pm$ %.2f' % (data[key].mean(), data[key].std()) for key in keys), '\\\\')

