"""
  @Time : 2022/3/14 16:07 
  @Author : Ziqi Wang
  @File : simulate.py 
"""
import sys
sys.path.append('../')

import json
import time
import argparse
from src.utils.filesys import get_path
from src.utils.parallel import AsyncPoolWithLoad, TaskLoad
from smb import MarioProxy, traverse_batched_level_files, MarioLevel

proxy = MarioProxy()
def simulate_single(lvlstr):
    level = MarioLevel(lvlstr)
    simlt_res = proxy.simulate_long(level)
    return simlt_res['trace'], MarioProxy.get_seg_infos(simlt_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='exp_data/randlvls/l5')
    parser.add_argument('--n_parallel', type=int, default=6)

    args = parser.parse_args()
    simulation_pool = AsyncPoolWithLoad(args.n_parallel)

    lvls_to_be_simlt = [*traverse_batched_level_files(args.path)]
    n = sum(len(item[0]) for item in lvls_to_be_simlt)
    results = []
    length_dict = {}
    for lvls, name in lvls_to_be_simlt:
        load_iter = (TaskLoad(str(lvl), name=name, index=i) for i, lvl in enumerate(lvls))
        simulation_pool.push(simulate_single, *load_iter)
        length_dict[name] = len(lvls)
        while simulation_pool.get_num_waiting() > 2 * args.n_parallel:
            results += simulation_pool.collect()
            time.sleep(1)
            print(f'{len(results)}/{n} finished')

    results += simulation_pool.wait_and_get()
    simulation_pool.close()
    print(f'Finish all simulation for {args.path}')

    res_dict = {}
    for res, info in results:
        name = info['name']
        index = info['index']
        if name not in res_dict.keys():
            res_dict[name] = [None] * length_dict[name]
        full_trace, seg_infos = res
        res_dict[name][index] = {'full_trace': full_trace, 'seg_infos': seg_infos}

    for name, value in res_dict.items():
        with open(get_path(f'{args.path}/{name}_simlt_res.json'), 'w') as f:
            json.dump(value, f)

