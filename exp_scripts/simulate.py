"""
  @Time : 2022/3/14 16:07 
  @Author : Ziqi Wang
  @File : simulate.py 
"""
import sys
sys.path.append('../')

import json
import argparse
from multiprocessing import Pool
from src.utils.filesys import get_path
from smb import MarioProxy, traverse_level_files, MarioLevel, MarioJavaAgents

proxy = MarioProxy()
def simulate_single(lvlstr):
    level = MarioLevel(lvlstr)
    simlt_res = proxy.simulate_long(level)
    return simlt_res['trace'], MarioProxy.get_seg_infos(simlt_res)

def simulate_vary_player(lvlstr):
    level = MarioLevel(lvlstr)
    simlt_res = proxy.simulate_long(level, MarioJavaAgents.Runner)
    res_r = {'full_trace': simlt_res['trace'], 'seg_infos': MarioProxy.get_seg_infos(simlt_res)}
    simlt_res = proxy.simulate_long(level, MarioJavaAgents.Killer, 10.)
    res_k = {'full_trace': simlt_res['trace'], 'seg_infos': MarioProxy.get_seg_infos(simlt_res)}
    simlt_res = proxy.simulate_long(level, MarioJavaAgents.Collector, 10.)
    res_c = {'full_trace': simlt_res['trace'], 'seg_infos': MarioProxy.get_seg_infos(simlt_res)}
    print('simulation finish')
    return res_r, res_k, res_c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--n_parallel', type=int, default=6)
    parser.add_argument('--vary_player', action='store_true')

    args = parser.parse_args()

    results = []
    pool = Pool(processes=args.n_parallel)
    for lvl, name in traverse_level_files(args.path):
        if args.vary_player:
            res = pool.apply_async(simulate_vary_player, (str(lvl),))
        else:
            res = pool.apply_async(simulate_single, (str(lvl),))
        results.append((name, res))

    pool.close()
    pool.join()
    n = 0
    for name, res in results:
        res.wait()
        if args.vary_player:
            res_runner, res_killer, res_collector = res.get()
            with open(get_path(f'{args.path}/{name}_runner_simlt_res.json'), 'w') as f:
                json.dump(res_runner, f)
            with open(get_path(f'{args.path}/{name}_killer_simlt_res.json'), 'w') as f:
                json.dump(res_killer, f)
            with open(get_path(f'{args.path}/{name}_collector_simlt_res.json'), 'w') as f:
                json.dump(res_collector, f)
        else:
            with open(get_path(f'{args.path}/{name}_simlt_res.json'), 'w') as f:
                full_trace, seg_infos = res.get()
                json.dump({'full_trace': full_trace, 'seg_infos': seg_infos}, f)

    print(f'Finish simulation for {args.path}')
