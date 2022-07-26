"""
  @Time : 2022/2/6 20:13 
  @Author : Ziqi Wang
  @File : genlvl_statistics.py
"""
import json
import os

import numpy as np

from smb import MarioLevel
from src.designer.use_designer import Designer
from src.environment.env import make_vec_offrew_env
from src.gan.gan_config import nz
from src.utils.filesys import get_path
from src.environment.rfuncs import default


def generate_levels(src_path, additional_folder='', save=True, n=1, l=25, n_parallel=1):
    designer = Designer(src_path + '/actor.pth')
    with open(get_path(f'{src_path}/kwargs.json'), 'r') as f:
        hist_len = json.load(f)['hist_len']
    env = make_vec_offrew_env(n_parallel, hist_len=hist_len, eplen=l, return_lvl=True)
    levels = []
    obs = env.reset()
    while len(levels) < n:
        actions = designer.act(obs)
        # actions, _ = designer.act(obs)
        next_obs, _, dones, infos = env.step(actions)
        del obs
        obs = next_obs
        for done, info in zip(dones, infos):
            if done:
                level = MarioLevel(info['LevelStr'])
                levels.append(level)
                print(len(levels))
    add_path = '' if additional_folder == '' else '/' + additional_folder
    os.makedirs(get_path(f'{src_path}{add_path}'), exist_ok=True)
    if save:
        for i in range(n):
            level = levels[i]
            level.save(f'{src_path}{add_path}/sample{i}')
        # if save_img:
        #     level.to_img(f'{src_path}{add_path}/sample{i}.png')
    return levels[:n]
    pass

def test(src_path, rfunc, n=30, l=25, n_parallel=6):
    if src_path == 'exp_data/randgen':
        designer = None
        hist_len = 1
    else:
        designer = Designer(src_path + '/actor.pth')
        with open(get_path(f'{src_path}/kwargs.json'), 'r') as f:
            hist_len = json.load(f)['hist_len']
    env = make_vec_offrew_env(n_parallel, rfunc, hist_len=hist_len, eplen=l)
    keys = [term.__class__.__name__ for term in rfunc.terms]
    # results = {key: [] for key in keys}
    results = []
    obs = env.reset()
    while len(results) < n:
        if designer is None:
            actions = np.random.rand(*obs.shape) * 2 - 1
        else:
            actions = designer.act(obs)
        # actions, _ = designer.act(obs)
        next_obs, _, dones, infos = env.step(actions)
        del obs
        obs = next_obs
        for done, info in zip(dones, infos):
            if done:
                results.append({key: info[key] for key in keys})
                # for key in keys:
                #     results[key].append(info[key])
    # add_path = '' if additional_folder == '' else '/' + additional_folder
    # os.makedirs(get_path(f'{src_path}{add_path}'), exist_ok=True)
    with open(get_path(f'{src_path}/test_rewards.json'), 'w') as f:
        res = {key: [results[i][key] for i in range(n)] for key in keys}
        json.dump(res, f)


if __name__ == '__main__':
    # test('exp_data/main/both', default, n=100)
    # test('exp_data/main/content', default, n=100)
    test('exp_data/main/killer', default, n=100)
    # test('exp_data/randgen', default, n=100)
    # generate_levels('exp_data/edrl-unigan/fb_p', 'samples', n=6, n_parallel=6)
    # generate_levels('exp_data/edrl-unigan/fc_p', 'samples', n=6, n_parallel=6)
    # generate_levels('exp_data/edrl-unigan/fc_fb_p', 'samples', n=6, n_parallel=6)

    # generate_levels('exp_data/edrl-unigan/fb_p', 'test', n=100, n_parallel=5)
    # generate_levels('exp_data/edrl-unigan/fc_p', 'test', n=100, n_parallel=5)
    # generate_levels('exp_data/edrl-unigan/fc_fb_p', 'test', n=100, n_parallel=5)
