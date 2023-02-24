"""
  @Time : 2022/3/14 16:21 
  @Author : Ziqi Wang
  @File : generate.py
"""
import json
import sys
sys.path.append('../')

import argparse
from src.utils.datastruct import batched_iter, RingQueue
from src.utils.filesys import get_path
from src.analyze.genlvl_statistics import generate_levels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--folder', type=str, default='samples')
    parser.add_argument('--n_init', type=int, default=100)
    parser.add_argument('--n_parallel', type=int, default=6)
    parser.add_argument('--l', type=int, default=50)
    parser.add_argument('--save_batch', type=int, default=100)

    args = parser.parse_args()

    if args.save_batch == 1:
        generate_levels(args.src_path, args.folder, n=args.n, l=args.l, n_parallel=args.n_parallel)
    else:
        levels = generate_levels(args.src_path, args.folder, n=args.n, l=args.l, n_parallel=args.n_parallel, save=False)
        level_strs = [str(level) for level in levels]
        for i, (batch, _) in enumerate(batched_iter(level_strs, args.save_batch)):
            file_content = ';\n'.join(batch)
            with open(get_path(f'{args.src_path}/{args.folder}/batch{i}.smblvs'), 'w') as f:
                f.write(file_content)

