"""
  @Time : 2022/1/4 10:03 
  @Author : Ziqi Wang
  @File : rfuncs.py 
"""

from src.environment.reward_function import *


default = RewardFunc(FunContent(), FunBehaviour(), Playability())
fun_c = RewardFunc(FunContent(), Playability())
fun_b = RewardFunc(FunBehaviour(), Playability())
fun_cb = RewardFunc(FunContent(), FunBehaviour(), Playability())
