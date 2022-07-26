# """
#   @Time : 2021/11/7 21:16
#   @Author : Ziqi Wang
#   @File : playability.py
# """
#
# from src.environment.reward_func import RewardFuncTerm
#
#
# class CompletingRatioReward(RewardFuncTerm):
#     def __init__(self, magnitude=1):
#         super(CompletingRatioReward, self).__init__(magnitude, True)
#         self.counter = 0
#
#     def compute_reward(self, **kwargs):
#         self.counter += 1
#         simlt_res = kwargs['simlt_res']
#         if simlt_res :
#             revised_ratio = (
#                 (simlt_res['completing-ratio'] - 2 / (self.counter + 2)) /
#                 (self.counter / (self.counter + 2))
#             )
#             reward = self.counter * revised_ratio
#             self.counter = 0
#             return reward
#         return 0
#
#
# class UnplayablePenalty(RewardFuncTerm):
#     def __init__(self, magnitude=1):
#         super(UnplayablePenalty, self).__init__(magnitude, True)
#         self.counter = 0
#
#     def compute_reward(self, **kwargs):
#         self.counter += 1
#         simlt_res = kwargs['simlt_res']
#         penalty = 0
#         if simlt_res:
#             if simlt_res['status'] != 'WIN':
#                 penalty -= self.counter
#             self.counter = 0
#         return penalty
#
