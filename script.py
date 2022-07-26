"""
  @Time : 2022/1/4 20:58 
  @Author : Ziqi Wang
  @File : script.py 
"""
import pygame

from smb import MarioLevel, level_sum, traverse_level_files, MarioProxy
from src.designer.use_designer import Designer
from src.utils.filesys import get_path
from src.utils.img import vsplit_img

if __name__ == '__main__':
    # path = 'edrl-unigan/fc_p'
    # proxy = MarioProxy()
    #
    # split_w = MarioLevel.tex_size * MarioLevel.default_seg_width
    # for lvl, name in traverse_level_files(f'exp_data/{path}/samples'):
    #     tmp = lvl[:, :168]
    #     img = tmp.to_img(f'exp_data/{path}/samples/{name}.png')
    #     trace = proxy.simulate_long(tmp)['trace']
    #     pygame.draw.lines(img, 'black', False, [(x, y-8) for x, y in trace], 3)
    #     vsplit_img(img, split_w, save_path=f'exp_data/{path}/samples/{name}_with_trace.png')

    # img = pygame.image.load(get_path('img/mapsheet.png'))
    # Btex = img.subsurface(48., 0., 16., 16.)
    # btex = img.subsurface(64., 0., 16., 16.)
    # pygame.image.save(Btex, get_path('assets/tile-11.png'))
    # pygame.image.save(btex, get_path('assets/tile-12.png'))
    for lvl, name in traverse_level_files(f'levels/MatthewDataset'):
        lvl.to_img(f'levels/MatthewDataset/{name}.png')

