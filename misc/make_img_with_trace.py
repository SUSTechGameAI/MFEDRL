"""
  @Time : 2022/2/12 11:45 
  @Author : Ziqi Wang
  @File : make_img_with_trace.py 
"""

from smb import MarioLevel, MarioProxy


if __name__ == '__main__':
    # src_path = './div_content'
    proxy = MarioProxy()
    lvl1 = MarioLevel.from_file('misc/div_behavior/1-1.gmtxt')
    print(lvl1)
    # lvl2 = MarioLevel.from_txt('misc/div_behavior/1-2.gmtxt')

    trace1 = proxy.simulate_game(lvl1, render=True)
    # trace2 = proxy.simulate_game(lvl2)
    pass
