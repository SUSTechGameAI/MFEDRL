"""
  @Time : 2022/3/29 16:41 
  @Author : Ziqi Wang
  @File : make_learning_curve.py 
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from src.utils.filesys import get_path


if __name__ == '__main__':
    with open(get_path('exp_data/main/content/ep_infos.json'), 'r') as f:
        data = json.load(f)
    # ttlist = [item['TotalScore'] for item in data]
    # fblist = [item['FunBehaviour'] for item in data]
    fclist = [item['FunContent'] for item in data]
    plist = [item['Playability'] for item in data]
    t = list(range(5000, 1000001, 5000))
    # ttmean = [(np.array(ttlist[s*200:(s+1)*200]) / 25).mean() for s in range(0, 200)]
    # fbmean = [(np.array(fblist[s*200:(s+1)*200]) / 25).mean() for s in range(0, 200)]
    fcmean = [(np.array(fclist[s*200:(s+1)*200]) / 25).mean() for s in range(0, 200)]
    pmean = [(np.array(plist[s*200:(s+1)*200]) / 25).mean() for s in range(0, 200)]
    # i = 0
    fig = plt.figure(figsize=(5, 2.5), dpi=320)
    # ax = fig.add_subplot(111)
    # ax2 = ax.twinx()
    # plt.plot(t, fbmean, label='$R_G$', color='red')
    plt.plot(t, fcmean, label='$R_L$', color='blue')
    plt.plot(t, pmean, label='$R_p$', color='green')
    # plt.plot(t, ttmean, label='$R_G+R_L+R_p$', color='black')
    # plt.title('Designer trained with $R_{fb} + R_p$')
    plt.xlabel('time steps', fontsize=12)
    plt.ylabel('reward per step', fontsize=12)
    plt.xticks([0, 2e5, 4e5, 6e5, 8e5, 1e6], ['0', '2e5', '4e5', '6e5', '8e5', '1e6'])
    plt.ylim((-0.6, 1))
    plt.legend(fontsize=11, ncol=3, loc='lower center')
    plt.grid()
    plt.tight_layout()
    plt.show(bbox_inches='tight')
    # while i < len(fblist):

    # ax.plot(time, Swdown, '-', label='Swdown')
    # ax.plot(time, Rn, '-', label='Rn')
    # ax2.plot(time, temp, '-r', label='temp')
    # ax.legend(loc=0)
    # ax.grid()
    # ax.set_xlabel("Time (h)")
    # ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
    # ax2.set_ylabel(r"Temperature ($^\circ$C)")
    # ax2.set_ylim(0, 35)
    # ax.set_ylim(-20, 100)
    # ax2.legend(loc=0)
    # plt.savefig('0.png')
    #     # pass
    pass

