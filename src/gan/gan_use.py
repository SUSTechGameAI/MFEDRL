"""
  @Time : 2022/1/27 15:20 
  @Author : Ziqi Wang
  @File : gan_use.py 
"""

import torch
from smb import MarioLevel
# from src.gan.gan_models import Generator
from src.gan.gan_config import nz
from src.utils.filesys import get_path
# from src.utils.img import make_img_sheet


def sample_latvec(n=1, span=2, device='cpu'):
    return torch.rand(n, nz, 1, 1, device=device) * span - span / 2

def process_levels(raw_tensor_lvls, to_lvl_obj=False):
    H, W = MarioLevel.height, MarioLevel.default_seg_width
    res = []
    for single in raw_tensor_lvls:
        lvl = single[:, :H, :W].detach().cpu().numpy()
        if to_lvl_obj:
            lvl = MarioLevel.from_one_hot_arr(lvl)
        res.append(lvl)
    return res

def get_generator(path='models/generator.pth', device='cpu'):
    safe_path = get_path(path)
    generator = torch.load(safe_path, map_location=device)
    generator.requires_grad_(False)
    generator.eval()
    return generator
    pass

# if __name__ == '__main__':
#     noise = sample_latvec(90)
#     generator = get_generator('exp_data/generator.pth')
#     levels = process_levels(generator(noise), True)
#     imgs = [lvl.to_img(None) for lvl in levels]
#     make_img_sheet(imgs, 10, save_path='exp_data/generator_samples/gaussian_sample_sheet.png')


