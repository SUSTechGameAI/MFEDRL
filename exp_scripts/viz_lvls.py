"""
  @Time : 2022/3/14 16:54 
  @Author : Ziqi Wang
  @File : viz_lvls.py 
"""

import sys
sys.path.append('../')

import json
import pygame
import argparse
from src.utils.filesys import get_path
from src.utils.img import make_img_sheet
from smb import MarioLevel, traverse_level_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--with_trace', action='store_true')

    args = parser.parse_args()

    h, w = MarioLevel.height, MarioLevel.default_seg_width
    for lvl, name in traverse_level_files(f'{args.path}'):
        s = args.start * w
        e = lvl.w if args.end == -1 else args.end * w
        tmp = lvl[:, s:e]
        img = tmp.to_img(f'{args.path}/{name}.png')
        if not args.with_trace:
            continue
        with open(get_path(f'{args.path}/{name}_simlt_res.json'), 'r') as f:
            full_trace = json.load(f)['full_trace']
        ts = MarioLevel.tex_size
        full_img = lvl.to_img(None)
        pygame.draw.lines(full_img, 'black', False, [(x, y-8) for x, y in full_trace], 3)
        pygame.image.save(
            full_img.subsurface((
                w * args.start * ts, 0, w * (args.end - args.start) * ts, h * ts,
            )),
            get_path(f'{args.path}/{name}_with_trace.png')
        )
        # imgs = [
        #     full_img.subsurface((ts * i, 0, w * ts, h * ts))
        #     for i in range(s, e, w)
        # ]
        # make_img_sheet(imgs, ncols=len(imgs), save_path=f'{args.path}/{name}_with_trace.png')
#