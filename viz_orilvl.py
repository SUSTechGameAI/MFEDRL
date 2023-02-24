"""
  @Time : 2022/3/31 11:06 
  @Author : Ziqi Wang
  @File : viz_orilvl.py 
"""
import json
from smb import traverse_level_files
from src.environment.reward_function import FunContent, FunBehaviour
from src.utils.filesys import get_path

if __name__ == '__main__':
    fun_c_func = FunContent(g=0.1, n=4)
    fun_b_func = FunBehaviour(g=0.25, n=4)
    for lvl, name in traverse_level_files('exp_data/orilvls'):
        with open(get_path(f'exp_data/orilvls/{name}_simlt_res.json'), 'r') as f:
            data = json.load(f)
        trace = data['full_trace']
        simlt_res = data['seg_infos'][:5]
        lvl[:, :28*5].to_img_with_trace(trace, f'exp_analysis/orilvl_analyze/{name}_with_trace.png')
        segs = [lvl[:, i*28:(i+1)*28] for i in range(5)]
        fc = sum(fun_c_func.compute_rewards(segs=segs)) / 4
        fb = sum(fun_b_func.compute_rewards(segs=segs, simlt_res=simlt_res)) / 4
        with open(get_path(f'exp_analysis/orilvl_analyze/{name}_fun.json'), 'w') as f:
            json.dump({'fc': fc, 'fb': fb}, f)

        pass


