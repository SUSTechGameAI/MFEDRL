"""
  @Time : 2022/1/1 16:03 
  @Author : Ziqi Wang
  @File : test.py 
"""
import random
import re
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import numpy as np
import openpyxl.utils.cell
from scipy.stats import wasserstein_distance

if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 10), dpi=192)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    # fig.text(0.04, 0.5, 'fun-content', rotation='vertical', va='center', ha='center')
    # fig.text(0.5, 0.04, 'fun-behavior', va='center', ha='center')
    for i in range(1, 26):
        ax = fig.add_subplot(5, 5, i)
        x1 = [random.random() for _ in range(100)]
        y1 = [random.random() for _ in range(100)]
        x2 = [(random.random() + 1) / 2 for _ in range(50)]
        y2 = [(random.random() + 1) / 2 for _ in range(50)]
        if i <= 5:
            plt.title('gc=1')
        if i % 5 == 0:
            plt.xlabel('gb=1', horizontalalignment='right')
        if not i % 5 == 1:
            plt.yticks([], [])
        if not (i - 1) // 5 == 4:
            plt.xticks([], [])
        plt.scatter(x1, y1, color='red', alpha=0.2)
        plt.scatter(x2, y2, color='blue', alpha=0.2)

    plt.show()

