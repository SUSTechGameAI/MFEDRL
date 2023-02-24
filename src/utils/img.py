"""
  @Time : 2022/3/10 11:22 
  @Author : Ziqi Wang
  @File : img.py 
"""

import pygame.image
from math import ceil

from smb import MarioLevel
from src.utils.filesys import get_path


def vsplit_img(img, w=MarioLevel.default_seg_width * MarioLevel.tex_size, lw=6, lcolor='white', save_path='./image.png'):
    n = ceil(img.get_width() / w)
    h = img.get_height()
    canvas = pygame.Surface((img.get_width() + (n-1) * lw, h))
    canvas.fill(lcolor)
    for i in range(n):
        canvas.blit(img.subsurface(i*w, 0, min(w, img.get_width()-i*w), h), (i*(w + lw), 0))
    pygame.image.save(canvas, get_path(save_path))
    pass


def make_img_sheet(imgs, ncols, x_space=6, y_space=6, space_color='white', save_path='./image.png'):
    nrows = ceil(len(imgs) / ncols)
    w, h = imgs[0].get_size()

    canvas = pygame.Surface((
        (w + x_space) * ncols - x_space, (h + y_space) * nrows - y_space
    ))
    canvas.fill(space_color)
    for i in range(len(imgs)):
        row_id, col_id = i // ncols, i % ncols
        canvas.blit(imgs[i], ((w + x_space) * col_id, (h + y_space) * row_id))

    pygame.image.save(canvas, get_path(save_path))

def stack_imgs(imgs, space=6, space_color='white', save_path='./image.png'):
    make_img_sheet(imgs, 1, y_space=space, space_color=space_color, save_path=save_path)

