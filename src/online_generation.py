# """
#   @Time : 2022/1/4 19:32
#   @Author : Ziqi Wang
#   @File : online_generation.py
# """
#
# import numpy as np
# from smb import MarioProxy
# from src.gan.gan_use import *
# from src.utils import RingQueue
#
#
# class GenerationTrial:
#     def __init__(self, designer, resample=False):
#         self.designer = designer
#         self.resample = resample
#         # self.simltor = MarioProxy()
#         self.simltor = None if not resample else MarioProxy()
#         self.onehot_lvl = None
#         self.generator = get_generator()
#         self.archive_len = self.designer.obs_len
#         # self.level_archive = RingQueue(self.archive_len)
#         self.latvec_archive = RingQueue(self.archive_len)
#         self.levels = []
#         pass
#
#     def start(self, l=100):
#         obs0 = np.random.randn(nz).astype(np.float32).clip(-1, 1)
#         self.latvec_archive.push(obs0)
#         # self.level_archive.push(seg0)
#         for _ in range(l):
#             self.step()
#         return self.levels
#
#     def step(self):
#         action = self.designer.predict(self.__get_obs())[0]
#         onehot_lvl = self.generator(torch.tensor(action).view(1, -1, 1, 1))
#         seg = process_levels(onehot_lvl, True)[0]
#         self.latvec_archive.push(action)
#         # self.level_archive.push(seg)
#         self.levels.append(seg)
#         pass
#
#     def __get_obs(self):
#         lack = self.archive_len - len(self.latvec_archive)
#         pad = [np.zeros([nz], np.float32) for _ in range(lack)]
#         return np.concatenate([*pad, *self.latvec_archive.to_list()])
#
#
