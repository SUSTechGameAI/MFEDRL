# """
#   @Time : 2021/9/27 21:11
#   @Author : Ziqi Wang
#   @File : fun.py
# """
#
# from abc import abstractmethod
# from smb import MarioLevel, level_sum
# from src.environment.reward_func import RewardFuncTerm
# from src.level_divs import tile_pattern_kl_div, ContentDivergenceMetrics
# from src.utils import a_clip, RingQueue
# from config import history_len
#
#
# class SlackingA_ClippedDivReward(RewardFuncTerm):
#     default_kwargs = {'goal_div': 0.15, 'novelty_metric': ContentDivergenceMetrics.TilePttrJS2}
#
#     def __init__(self, magnitude=1, **kwargs):
#         cfgs = SlackingA_ClippedDivReward.default_kwargs.copy()
#         cfgs.update(kwargs)
#         super(SlackingA_ClippedDivReward, self).__init__(magnitude, False, **cfgs)
#         self.metric = cfgs['novelty_metric'].get_func()
#         self.g = cfgs['goal_div']
#
#     def compute_reward(self, **kwargs):
#         seg = kwargs['seg']
#         archive = kwargs['archive']
#         if not archive:
#             return 0
#         reward = 0
#         r_sum = 0
#         for k in range(1, len(archive) + 1):
#             i = len(archive) - k
#             cmp_seg = archive[i]
#             r = 1 - k / (history_len + 1)
#             r_sum += r
#             reward += a_clip(self.metric(seg, cmp_seg), self.g, r)
#         return reward / r_sum
#
#
# class OldFun(RewardFuncTerm):
#     default_kwargs = {'n': 3, 'ub': 0.94, 'lb': 0.26, 'delta': 7, 'novelty_metric': ContentDivergenceMetrics.TilePttrKL2}
#
#     def __init__(self, magnitude=1, **kwargs):
#         cfgs = OldFun.default_kwargs.copy()
#         cfgs.update(kwargs)
#         super(OldFun, self).__init__(magnitude, False, **cfgs)
#         self.metric = cfgs['novelty_metric'].get_func()
#         self.ub = cfgs['ub']
#         self.lb = cfgs['lb']
#         self.n = cfgs['n']
#         self.delta = cfgs['delta']
#
#     def compute_reward(self, **kwargs):
#         seg = kwargs['seg']
#         archive = kwargs['archive']
#         if not archive:
#             return 0
#         w = seg.w
#         history = level_sum([*archive, seg])
#
#         div_sum = 0.
#         for k in range(self.n + 1):
#             s, e = history.w - k * self.delta - w, history.w - k * self.delta
#             if s < 0:
#                 break
#             cmp_seg = history[:, s:e]
#             div_sum += self.metric(seg, cmp_seg)
#         div = div_sum / (self.n + 1)
#         return min(0., div - self.lb, self.ub - div) ** 2
#
#
# class HistoricalDeviation(RewardFuncTerm):
#     default_kwargs = {'m': 20, 'n': 10, 'delta': None, 'novelty_metric': ContentDivergenceMetrics.TilePttrKL2}
#
#     def __init__(self, magnitude=1, **kwargs):
#         cfgs = HistoricalDeviation.default_kwargs.copy()
#         cfgs.update(kwargs)
#         super(HistoricalDeviation, self).__init__(magnitude, False, **cfgs)
#         self.metric = cfgs['novelty_metric'].get_func()
#         self.m = cfgs['m']
#         self.n = cfgs['n']
#         self.delta = cfgs['delta']
#         self.archive = RingQueue(cfgs['m'])
#
#     def compute_reward(self, **kwargs):
#         seg = kwargs['seg']
#         if not len(self.archive):
#             self.archive.push(seg)
#             return 0
#         delta = seg.w if self.delta is None else self.delta
#         w = seg.w
#         history = level_sum(self.archive.to_list())
#         divs = []
#         for k in range(self.m):
#             s, e = history.w - k * delta - w, history.w - k * delta
#             if s < 0:
#                 break
#             cmp_seg = history[:, s:e]
#             div = self.metric(seg, cmp_seg)
#             divs.append(div)
#         divs.sort()
#         self.archive.push(seg)
#         return sum(divs[:self.n]) / self.n
#
#     def on_reset(self):
#         self.archive.clear()
#
#
# if __name__ == '__main__':
#     l1 = MarioLevel.from_txt('E:\\academic\my works\EDRL\EDPCGRL\src/generator/samples/seg0.txt')
#     l2 = MarioLevel.from_txt('E:\\academic\my works\EDRL\EDPCGRL\src/generator/samples/seg3.txt')
#     print(tile_pattern_kl_div(l1, l2, 2))
#     pass
