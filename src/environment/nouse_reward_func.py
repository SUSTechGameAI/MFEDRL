# """
#   @Time : 2021/9/16 18:53
#   @Author : Ziqi Wang
#   @File : reward_func.py
# """
#
# import numpy as np
# from abc import abstractmethod, ABC
# from src.utils import tab_per_line
#
#
# class RewardFunction:
#     def __init__(self, *args):
#         if len(set(term.__class__.__name__ for term in args)) != len(args):
#             raise RuntimeError('Reward function cannot contain multiple terms of same type')
#         self.terms = args
#         self.require_simlt = False
#         for term in self.terms:
#             self.require_simlt = self.require_simlt or term.require_simlt
#
#     def __call__(self, **kwargs):
#         reward = sum(func(**kwargs) for func in self.terms)
#         return reward
#
#     def __str__(self):
#         res = 'Reward Function {\n' + ',\n'.join(map(str, self.terms)) + '\n}\n'
#         return res
#
#     def reset(self) -> dict:
#         term_rewards = {}
#         for term in self.terms:
#             term_rewards[term.__class__.__name__] = term.reset()
#         return term_rewards
#
#
# class RewardFuncTerm:
#     def __init__(self, magnitude, require_simlt, **kwargs):
#         self.magnitude = magnitude
#         self.info = 0
#         arg_text = (
#             '\n\t' + ',\n\t'.join(f'{key}={val}' for key, val in kwargs.items()) + '\n'
#             if kwargs else ''
#         )
#         self.init_log = f'{magnitude} * {self.__class__.__name__}(%s)' % arg_text
#         self.init_log = tab_per_line(self.init_log)
#         self.require_simlt = require_simlt
#
#     def __call__(self, **kwargs):
#         value = self.compute_reward(**kwargs)
#         self.info += value
#         return self.magnitude * value
#
#     def __str__(self):
#         return self.init_log
#
#     @abstractmethod
#     def compute_reward(self, **kwargs):
#         pass
#
#     def on_reset(self):
#         pass
#
#     def reset(self):
#         information = self.info
#         self.info = 0
#         self.on_reset()
#         return information
#
