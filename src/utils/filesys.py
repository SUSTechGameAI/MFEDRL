"""
  @Time : 2022/3/10 11:16 
  @Author : Ziqi Wang
  @File : filesys.py 
"""

import os
from root import PRJROOT


def auto_dire(path=None, name='trial'):
    dire_id = 0
    prefix = PRJROOT if path is None else f'{get_path(path)}/'
    tar = f'{prefix}'
    while os.path.exists(tar):
        tar = f'{prefix}{name}{dire_id}'
        dire_id += 1
    os.makedirs(tar)
    return tar


def get_path(path):
    """ if is absolute path or working path(./, .\\), return {path}, else return {PRJROOT + path} """
    if os.path.isabs(path) or path[:2] in {'./', '.\\'}:
        return path
    else:
        return PRJROOT + path