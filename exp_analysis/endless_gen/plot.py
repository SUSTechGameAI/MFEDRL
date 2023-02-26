"""
  @Time : 2022/9/30 14:18 
  @Author : Ziqi Wang
  @File : plot.py 
"""

import numpy as np
import matplotlib.pyplot as plt
from smb import load_batch
from itertools import product
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.utils.img import vsplit_img
from src.utils.filesys import get_path


def plot_step_rewards(rewards, md_divs, td_divs, save_path='', title=''):
    n, h = rewards.shape
    x = list(range(1, h + 1))
    y = np.array([rewards[:, i].mean() for i in range(h)])
    std = np.array([rewards[:, i].std() for i in range(h)])
    plt.style.use('default')

    plt.figure(figsize=(4, 2), dpi=384)
    ax = plt.axes()

    ax.plot([h // 2, h // 2], [0, 2.3], color='gray', ls='--', lw=1)
    ax.plot(x, y, color='blue')
    ax.fill_between(x, y-std, y+std, color='blue', alpha=0.2, linewidth=0)
    ax.set_ylim((0, 2.3))
    ax.set_ylabel('Average reward', color='blue')
    ax.set_yticks([0, 1, 2], ['0', '1', '2'], color='blue')
    ax.set_xlabel('Generation step')
    ax2 = ax.twinx()
    # ax2.plot([0, *x], 5 * md_divs, color='red')
    # print(td_divs.shape)
    ax2.plot([0, *x], td_divs, color='red')
    ax2.set_ylim((0, 0.48))
    ax2.set_ylabel('Diversity', color='red')
    ax2.set_yticks([0, 0.2, 0.4], ['0.0', '0.2', '0.4'], color='red')

    # plt.xlabel('Time step')
    plt.title(title)
    plt.tight_layout(pad=0.2)
    if save_path:
        plt.savefig(get_path(save_path))
    else:
        plt.show()
    pass

def state_half_partition(ztraces, n):
    k, T, _ = ztraces.shape
    h = (T - n) // 2
    s0 = np.concatenate([
        np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
        for e in range(n, 2*n)
    ], axis=0)
    s1 = np.concatenate([
        np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
        for e in range(2*n, n+h)
    ], axis=0)
    s2 = np.concatenate([
        np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
        for e in range(n+h, T)
    ], axis=0)
    # print(s0.shape, s1.shape, s2.shape)
    return s0, s1, s2
    # s0 = np.concatenate([
    #     np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
    #     for e in range(n, 2*n)
    # ], axis=0)
    # s1 = np.concatenate([
    #     np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
    #     for e in range(2*n, n+10)
    # ], axis=0)
    # s2 = np.concatenate([
    #     np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
    #     for e in range(n+10, n+h)
    # ], axis=0)
    # s3 = np.concatenate([
    #     np.concatenate([ztraces[:, i] for i in range(e-n, e)], axis=-1)
    #     for e in range(n+h, T)
    # ], axis=0)
    # # print(s0.shape, s1.shape, s2.shape)
    # return s0, s1, s2, s3

def plot_compression_scatter(sets, labels, compression='t-SNE', save_path='', colors=None, title=''):
    if colors is None:
        colors = [None] * len(sets)
    x = np.concatenate(sets, axis=0)
    splits = [0]
    for i in range(len(sets)):
        splits.append(splits[i] + len(sets[i]))

    if compression == 't-SNE':
        ts = TSNE(n_components=2, init='pca', learning_rate='auto')
        embx = np.array(ts.fit_transform(x))
    else:
        pca = PCA(2)
        embx = np.array(pca.fit_transform(x))
    print(splits)
    embs = [embx[splits[i]:splits[i+1]] for i in range(len(sets))]
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(2.5, 2.25), dpi=384)
    for emb, lb, c in zip(embs, labels,colors):
        plt.scatter(emb[:,0], emb[:,1], c=c, label=lb, alpha=0.15, linewidths=0, s=7)

    xspan = 1.2 * max(abs(embx[:, 0].max()), abs(embx[:, 0].min()))
    yspan = 1.05 * max(abs(embx[:, 1].max()), abs(embx[:, 1].min()))
    plt.xlim([-xspan, xspan])
    plt.ylim([-yspan, yspan])
    plt.xticks([-xspan, -0.5 * xspan, 0, 0.5 * xspan, xspan], [''] * 5)
    plt.yticks([-yspan, -0.5 * yspan, 0, 0.5 * yspan, yspan], [''] * 5)
    plt.title(title)
    # plt.legend()
    plt.tight_layout(pad=0.2)
    if save_path:
        plt.savefig(get_path(save_path))
    else:
        plt.show()

def vis_lvls():
    # for gm, n, i in product(('070', '099'), (4, 6), range(10)):
    #     lvl = load_batch(f'exp_analysis/endless_gen/data/gm{gm}n{n}.smblvs')[i]
    #     vsplit_img(
    #         lvl[:, n*28:(n+10)*28].to_img(), 28*16,
    #         save_path=f'exp_analysis/endless_gen/illustrations/gm{gm}n{n}/lvl-{i}.png'
    #     )

    for gm, n in product(('070', '080', '090', '099'), (4, 6)):
        lvl = load_batch('exp_analysis/endless_gen/data/gm070n4.smblvs')[0]
        img = lvl[:, n*28: (n+6)*28].to_img()
        vsplit_img(img, save_path=f'exp_analysis/endless_gen/illustrations/1-6/gm{gm}n{n}.png')
        img = lvl[:, (n+15)*28:(n+21)*28].to_img()
        vsplit_img(img, save_path=f'exp_analysis/endless_gen/illustrations/16-21/gm{gm}n{n}.png')
    pass


if __name__ == '__main__':
    # vis_lvls()
    # print(plt.style.available)
    # n_init = history_len
    for gm, n_init in product(('070', '080', '090', '099'), (4, 6)):
        gmtxt = gm[0] + '.' + gm[1:]
        # r = np.load(get_path(f'exp_analysis/endless_gen/data2/gm{gm}n{n_init}_rewards.npy'))
        # mdd = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{n_init}_md_divs.npy'))
        # tdd = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{n_init}_td_divs.npy'))
        # plot_step_rewards(
        #     r[:, -50:], mdd, tdd, f'exp_analysis/endless_gen/results/curves/gm{gm}n{n_init}.png',
        #     title=f'$\gamma={gmtxt}, n={n_init}$'
        # )
        ztraces = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{n_init}.npy'))
        # s0, s1, s2 = state_half_partition(ztraces, n_init)
        # plot_compression_scatter(
        #     [s0, s1, s2], ['initial', 'former', 'later'], colors=['black', 'red', 'blue'],
        #     save_path=f'exp_analysis/endless_gen/results/t-SNE/gm{gm}n{n_init}.png',
        #     title=f'$\gamma={gmtxt}, n={n_init}$'
        # )
        s0, s1, s2, s3 = state_half_partition(ztraces, n_init)
        plot_compression_scatter(
            [s0, s1, s2, s3], ['initial', 'early', 'former', 'later'], colors=['black', 'green', 'red', 'blue'],
            save_path=f'exp_analysis/endless_gen/results/t-SNE/gm{gm}n{n_init}-partition10.png',
            title=f'$\gamma={gmtxt}, n={n_init}$'
        )
    pass
