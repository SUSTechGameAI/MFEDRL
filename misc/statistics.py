"""
  @Time : 2022/3/16 13:45 
  @Author : Ziqi Wang
  @File : statistics.py 
"""

import json
import numpy as np
from src.utils.filesys import get_path

if __name__ == '__main__':
    with open(get_path('exp_data/edrl-unigan/hist7/fun_bc1/samples/evaluations.json'), 'r') as f:
        evaluations = json.load(f)
    nc = len(evaluations[0]['content_novelties'])
    nb = len(evaluations[0]['behavior_novelties'])
    cnovelties = np.array([item['content_novelties'] for item in evaluations])
    bnovelties = np.array([item['behavior_novelties'] for item in evaluations])
    print(cnovelties.mean(axis=0))
    print(bnovelties.mean(axis=0))
    pass

