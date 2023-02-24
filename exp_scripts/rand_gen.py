"""
  @Time : 2022/3/18 13:14 
  @Author : Ziqi Wang
  @File : gan_sample.py 
"""
import sys
import time

sys.path.append('../')

import os
import argparse
from src.gan.gan_use import *
from src.utils.parallel import MyAsyncPool
from src.utils.datastruct import batched_iter
from src.repairer.repairer import DivideConquerRepairer
from src.analyze.genlvl_statistics import generate_levels


repairer = DivideConquerRepairer()
def repair_func(lvlstr):
    return str(repairer.repair(MarioLevel(lvlstr)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str, default='models/generator.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--folder', type=str, default='samples')
    parser.add_argument('--n_init', type=int, default=200)
    parser.add_argument('--l', type=int, default=5)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--file_batch', type=int, default=100)
    parser.add_argument('--space_size', type=float, default=1.)
    parser.add_argument('--repair_parallel', type=int, default=5)
    parser.add_argument('--res_path', type=str, default='exp_data/randlvls')

    args = parser.parse_args()
    generator = get_generator(args.generator_path, args.device)

    repair_pool = MyAsyncPool(args.repair_parallel)

    n = 0
    lvl_strs = []
    while n < args.n:
        batch_size = min(args.n - n, args.batch)
        lvls = process_levels(generator(sample_latvec(batch_size, args.space_size * 2, args.device)), True)
        for _ in range(1, args.l):
            segs = process_levels(generator(sample_latvec(batch_size, args.space_size * 2, args.device)), True)
            for i, seg in enumerate(segs):
                lvls[i] = lvls[i] + seg
        n += batch_size
        lvl_strs += repair_pool.collect()
        for lvl in lvls:
            repair_pool.push(repair_func, (str(lvl),))
        while repair_pool.get_num_waiting() > 2 * args.repair_parallel:
            time.sleep(1)
            lvl_strs += repair_pool.collect()

    lvl_strs += repair_pool.wait_and_get()
    repair_pool.close()
    os.makedirs(get_path(args.res_path), exist_ok=True)
    for i, (batch, _) in enumerate(batched_iter(lvl_strs, args.file_batch)):
        file_content = ';\n'.join(batch)
        with open(get_path(f'{args.res_path}/batch{i}.smblvs'), 'w') as f:
            f.write(file_content)


    # while len(lvls) < args.n_init:
    #     batch_lvls = generator()
    # generate_levels(args.src_path, args.folder, n_init=args.n_init, l=args.l, n_parallel=args.n_parallel)
