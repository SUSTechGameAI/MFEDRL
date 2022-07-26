"""
  @Time : 2022/3/7 16:32 
  @Author : Ziqi Wang
  @File : rewterms.py 
"""

from abc import abstractmethod
from src.utils.mymath import a_clip
from src.environment.env_cfgs import history_len
from src.level_divs import trace_div, tile_pattern_js_div


class RewardFunc:
    def __init__(self, *args):
        self.terms = args
        self.require_simlt = any(term.require_simlt for term in self.terms)

    def get_rewards(self, **kwargs):
        return {
            term.get_name(): term.compute_rewards(**kwargs)
            for term in self.terms
        }

    def __str__(self):
        return 'Reward Function:\n' + ',\n'.join('\t' + str(term) for term in self.terms)


class RewardTerm:
    def __init__(self, require_simlt):
        self.require_simlt = require_simlt

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def compute_rewards(self, **kwargs):
        pass


class Playability(RewardTerm):
    def __init__(self, magnitude=1):
        super(Playability, self).__init__(True)
        self.magnitude=magnitude

    def compute_rewards(self, **kwargs):
        simlt_res = kwargs['simlt_res']
        return [0 if item['playable'] else -self.magnitude for item in simlt_res[1:]]

    def __str__(self):
        return f'{self.magnitude} * Playability'


class Fun(RewardTerm):
    def __init__(self, magnitude, goal_div, require_simlt, n, mode):
        super().__init__(require_simlt)
        self.g = goal_div
        self.magnitude = magnitude
        self.n = n
        self.mode = mode

    def compute_rewards(self, **kwargs):
        n_segs = len(kwargs['segs'])
        rewards = []
        for i in range(1, n_segs):
            reward = 0
            r_sum = 0
            for k in range(1, self.n + 1):
                j = i - k
                if j < 0:
                    break
                r = 1 - k / (self.n + 1)
                r_sum += r
                reward += a_clip(self.disim(i, j, **kwargs), self.g, r, self.mode)
            rewards.append(reward * self.magnitude / r_sum)
        return rewards

    @abstractmethod
    def disim(self, i, j, **kwargs):
        pass


class FunContent(Fun):
    def __init__(self, magnitude=1, g=0.1, w=2, n=history_len, mode=0):
        super(FunContent, self).__init__(magnitude, g, False, n, mode)
        self.w = w

    def disim(self, i, j, **kwargs):
        segs = kwargs['segs']
        seg1, seg2 = segs[i], segs[j]
        return tile_pattern_js_div(seg1, seg2, self.w)

    def __str__(self):
        return f'{self.magnitude} * FunContent(g={self.g:.3g}, w={self.w}, n={self.n})'


class FunBehaviour(Fun):
    def __init__(self, magnitude=1, g=0.25, w=10, n=history_len, mode=0):
        super(FunBehaviour, self).__init__(magnitude, g, True, n, mode)
        self.w = w

    def disim(self, i, j, **kwargs):
        simlt_res = kwargs['simlt_res']
        trace1, trace2 = simlt_res[i]['trace'], simlt_res[j]['trace']
        return trace_div(trace1, trace2, self.w)

    def __str__(self):
        return f'{self.magnitude} * FunBehavior(g={self.g:.3g}, w={self.w}, n={self.n})'

