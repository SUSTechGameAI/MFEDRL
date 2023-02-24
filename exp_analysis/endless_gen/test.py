"""
  @Time : 2022/9/27 12:51 
  @Author : Ziqi Wang
  @File : tests.py 
"""
import csv
import json
import time
from itertools import combinations, product

import numpy as np
from math import ceil

from exp_analysis.endless_gen.generate import generate_levels
from smb import MarioLevel, MarioProxy, load_batch
from multiprocessing import Pool

from src.designer.use_designer import Designer
from src.gan.gan_use import sample_latvec, get_generator, process_levels
from src.level_divs import tile_normalised_hamming
from src.utils.filesys import get_path
from src.environment.rfuncs import default as rfunc
from src.environment.env_cfgs import history_len
from src.utils.mymath import lpdist_mat
from dtw import dtw
from sklearn.cluster import KMeans


def evaluate_rewards(strlvl):
    start_time = time.time()
    lvl = MarioLevel(strlvl)
    proxy = MarioProxy()
    W = MarioLevel.default_seg_width
    segs = [lvl[:, s:s+W] for s in range(0, lvl.w, W)]
    simlt_res = MarioProxy.get_seg_infos(proxy.simulate_long(lvl))
    reward_lists = rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
    rewards = [sum(item) for item in zip(*reward_lists.values())]
    print('Evaluation finished in %.3fs' % (time.time() - start_time))
    return rewards[-(lvl.w // W - history_len):]
    pass

def evaluate_samples_rewards(src_path, dest_path, parallel=1):
    lvls = load_batch(src_path)
    pool = Pool(parallel)
    results = []
    for lvl in lvls:
        results.append(pool.apply_async(evaluate_rewards, (str(lvl),)))
    pool.close()
    pool.join()

    data = [res.get() for res in results]
    np.save(get_path(dest_path), data)
    pass

def get_z_targets(ztraces, n_init, dest_path):
    s = np.concatenate(ztraces[:, n_init:, :])
    kmeans = KMeans(10)
    kmeans.fit(s)
    if dest_path:
        np.save(get_path(dest_path), kmeans.cluster_centers_)
    return kmeans.cluster_centers_

def evaluate_step_diversity(designer, targets, dest_path, nt=30, ns=100):
    md_divs = np.zeros([nt, 51])
    td_divs = np.zeros([nt, 51])
    for i in range(nt):
        _, ztraces = generate_levels(designer, n=ns, h=50, parallel=25)
        ztraces = np.stack(ztraces)[:, history_len-1:, :]
        print(ztraces.shape)
        nt = len(ztraces)
        for t in range(51):
            s = ztraces[:, t, :]
            distab = lpdist_mat(s, s)
            md_divs[i, t] = distab.sum() / (nt * nt - nt)
            tar_distab = lpdist_mat(s, targets)
            td_divs[i, t] = np.min(tar_distab, axis=1).sum() / (len(targets) * nt)

    np.save(get_path(dest_path + '_md_divs.npy'), md_divs.mean(axis=0))
    np.save(get_path(dest_path + '_td_divs.npy'), td_divs.mean(axis=0))
    pass

def evaluate_z_diversity(ztraces, n_init):
    nt, h, _ = ztraces.shape
    h -= n_init
    distab = np.zeros([nt, nt])
    for i, j in product(range(nt), range(nt)):
        x = ztraces[i, -h:, :]
        y = ztraces[j, -h:, :]
        dist, *_ = dtw(x, y, lambda p, q: np.linalg.norm(p-q), 8)
        # dist, *_ = dtw(x, y, lambda p, q: np.linalg.norm(p-q), 8)
        distab[i, j] = dist
        distab[j, i] = dist
    return distab.sum() / (nt * nt - nt), np.min(distab + np.identity(nt) * 10000., axis=1).sum() / nt ** 0.5

def evaluate_lvl_diversity(src_path, n_init):
    lvls = load_batch(src_path)
    segss = [lvl.to_segs() for lvl in lvls]
    nt, h = len(lvls), len(segss[0])
    h -= n_init
    distab = np.zeros([nt, nt])
    for i, j in product(range(nt), range(nt)):
        x = segss[i][-h:]
        y = segss[j][-h:]
        dist, *_ = dtw(x, y, lambda p, q: tile_normalised_hamming(p, q), 3)
        # x = lvls[i][:, n_init*28:]
        # y = lvls[j][:, n_init*28:]
        # dist = tile_normalised_hamming(x, y)
        distab[i, j] = dist / h
        distab[j, i] = dist / h
    return distab.sum() / (nt * nt - nt)

def test_lvl_diversity(designer, nt=30):
    divs = []
    for i in range(nt):
        lvls, _ = generate_levels(designer, n=2, h=50, parallel=2)
        segs1 = lvls[0].to_segs()
        segs2 = lvls[1].to_segs()
        dist, *_ = dtw(segs1[-50:], segs2[-50:], lambda p, q: tile_normalised_hamming(p, q), w=history_len)
        divs.append(dist / 50)
    return sum(divs) / nt
    # np.save(get_path(dest_path + '_md_divs.npy'), md_divs.mean(axis=0))
    # np.save(get_path(dest_path + '_td_divs.npy'), td_divs.mean(axis=0))
    pass

def test_mean_rand_dist(nt=1000, dtw_w=history_len, h=50):
    generator = get_generator(device='cuda:0')
    dists = []
    for _ in range(nt):
        z1 = sample_latvec(h, device='cuda:0')
        z2 = sample_latvec(h, device='cuda:0')
        segs1 = process_levels(generator(z1), True)
        segs2 = process_levels(generator(z2), True)
        dist, c, d, pt = dtw(segs1, segs2, lambda p, q: tile_normalised_hamming(p, q), w=dtw_w)
        # print(c, d, pt)
        dists.append(dist / h)
    # with open(get_path('exp_analysis/endless_gen/rand_dist.json'), 'w') as f:
    #     json.dump(sum(dists) / len(dists), f)
    return sum(dists) / len(dists)
    pass


if __name__ == '__main__':
    # # ztraces = np.load(get_path('exp_analysis/endless_gen/data/gm070n4.npy'))
    # # print(ztraces.shape)
    # print(np.array([[1, 2, 3], [3, 2, 4]]).mean(axis=0))
    #
    # for gm, n in product(('070', '080', '090', '099'), (4, 6)):

    with open(get_path('exp_analysis/endless_gen/results/lvsdivs-dtw.csv'), 'a', newline='') as f:
        wrtr = csv.writer(f)
        wrtr.writerow(['gamma', 'n', 'diversity', ''])
        for gm in ('070', '080', '090', '099'):
            d = Designer(f'exp_data/recurrent_transition/n{history_len}/gm{gm}/actor.pth')
            # ztrcs = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{n}.npy'))
            # tgts = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{history_len}_targets.npy'))
            # get_z_targets(a, n, f'exp_analysis/endless_gen/data/gm{gm}n{n}_targets.npy')
            # evaluate_step_diversity(d, tgts, f'exp_analysis/endless_gen/data/gm{gm}n{history_len}')
            divs_dtw = test_lvl_diversity(d, 1000)
            wrtr.writerow([gm, history_len, divs_dtw, ''])
            print(gm, history_len, divs_dtw)
    # with open(get_path('exp_analysis/endless_gen/results/lvsdivs-dtw.csv'), 'w') as f:
    #     wrtr = csv.writer(f)
    #     wrtr.writerow(['gamma', 'n', 'diversity', ''])
    #     for gm, n in product(('070', '080', '090', '099'), (4, 6)):
    #         # a = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{n}.npy'))
    #         divs_dtw = evaluate_lvl_diversity(f'exp_analysis/endless_gen/data/gm{gm}n{n}.smblvs', n)
    #         # md_divs, td_divs = evaluate_lvl_diversity(a, n)
    #         # md_divs, td_divs = evaluate_seg_diversity(f'exp_analysis/endless_gen/data/gm{gm}n{n}.smblvs', n)
    #         wrtr.writerow([gm, n, divs_dtw, ''])
    #         print(gm, n, divs_dtw)
    # with open(get_path('exp_analysis/endless_gen/results/lvsdivs.csv'), 'w') as f:
    #     wrtr = csv.writer(f)
    #     wrtr.writerow(['gamma', 'n', 'diversity', ''])
    #     for gm, n in product(('070', '080', '090', '099'), (4, 6)):
    #         a = np.load(get_path(f'exp_analysis/endless_gen/data/gm{gm}n{n}.npy'))
    #         md_divs, td_divs = evaluate_level_diversity(a, n)
            # md_divs = evaluate_lvl_diversity(f'exp_analysis/endless_gen/data/gm{gm}n{n}.smblvs', n)
            # wrtr.writerow([gm, n, md_divs, ''])
            # print(gm, n, md_divs)

    # print(test_mean_rand_dist(1000, 4))
    # print(test_mean_rand_dist(1000, 6))
    pass
