"""
  @Time : 2022/3/10 11:15 
  @Author : Ziqi Wang
  @File : math.py 
"""
import numpy as np
from scipy.stats import entropy


def a_clip(v, g, r=1.0, mode=0):
    if mode == 0:
        return min(r, 1 - abs(v - g) / g)
    elif mode == 1:
        return min(r, v / g)
    elif mode == -1:
        return min(r, 2 - v / g)

def jsdiv(p, q):
    return (entropy(p, p + q, base=2) + entropy(q, p + q, base=2)) / 2

def grid_cnt(data, ranges, n_grids=10, normalize=True):
    eps = 1e-10
    d = data.shape[1]
    res = np.zeros([n_grids] * d)
    itvs = (ranges[:, 1] - ranges[:, 0]) * ((1 + eps) / n_grids)

    for item in data:
        indexes = tuple((item // itvs))
        res[indexes] = res[indexes] + 1
    if normalize:
        res /= res.size
    return res

def lpdist_mat(X, Y, p=2):
    diff = np.abs(np.expand_dims(X, axis=1) - np.expand_dims(Y, axis=0))
    distance_matrix = np.sum(diff ** p, axis=-1) ** (1 / p)
    return distance_matrix

if __name__ == '__main__':
    a = [[1, 2], [2, 3], [3, 4]]
    b = [[1.5, 2.5], [2.5, 3.5]]
    print(lpdist_mat(a, b))