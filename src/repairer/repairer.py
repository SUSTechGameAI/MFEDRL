"""
  @Time : 2022/1/5 20:29 
  @Author : Ziqi Wang
  @File : repairer.py 
"""

import numpy as np
import torch
import bisect
import itertools
import random
import time
from itertools import product
from smb import MarioLevel, level_sum
from src.repairer.cnet import get_cnet


class Detector:
    shifts = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    def __init__(self, cnet=None, theta1=0.1, theta2=0.5):
        self.model = get_cnet() if cnet is None else cnet
        self.model.eval()
        self.model.requires_grad_(False)
        self.theta1 = theta1
        self.theta2 = theta2

    @staticmethod
    def get_srrds_centers(padded_lvl, pos):
        surroundings, centers = [], []
        for i, j in pos:
            srrd_pos = np.array([i, j]) + Detector.shifts
            srrd = padded_lvl[srrd_pos[:, 0], srrd_pos[:, 1]]
            surroundings.append(srrd)
            centers.append(padded_lvl[i, j])
        return surroundings, centers

    @staticmethod
    def get_slt_space(padded_lvl):
        h, w = padded_lvl.shape
        ppset = MarioLevel.pipe_intset
        res = []
        for i, j in product(range(1, h-1), range(1, w-1)):
            indexes = np.array([i, j]) + Detector.shifts
            srrd = padded_lvl[indexes[:, 0], indexes[:, 1]]
            if padded_lvl[i, j] in ppset or any(x in ppset for x in srrd):  # only record combinations near pipes
                res.append([i, j])
        return np.array(res)

    def get_predictions(self, surroundings):
        x = torch.tensor(surroundings, dtype=torch.long)
        predictions = self.model(x).to('cpu').numpy()
        return predictions

    def get_probs(self, surroundings, centers):
        predictions = self.get_predictions(surroundings)
        probs = predictions[range(len(centers)), centers]
        return probs


class Individual:
    def __init__(self, solution):
        self.solution = solution
        self.fitness = None

    @staticmethod
    def rand_initial(problem):
        solution = problem.original.copy()
        return Individual(solution)

    def __getitem__(self, item):
        return self.solution[item]

    def decode(self, target_level, slt_space):
        target_level[slt_space[:, 0], slt_space[:, 1]] = self.solution


class Problem:
    def __init__(self, level, slt_space):
        # self.eval_space = eval_space
        self.slt_space = slt_space
        self.original = level[self.slt_space[:, 0], self.slt_space[:, 1]].copy()
        self.buffer = level.copy()
        pass

    def decode(self, idv):
        xs = self.slt_space[:, 0]
        ys = self.slt_space[:, 1]
        self.buffer[xs, ys] = idv.solution
        return self.buffer


class Repairer:
    def __init__(self, detector=None, pop_size=20, pm=0.1, pr=0.25, rrt=6):
        self.detector = Detector() if detector is None else detector
        self.pop_size = pop_size
        self.pm = pm
        self.pr = pr
        self.rrt = rrt
        pass

    def evaluate(self, idv, problem):
        phenotype = problem.decode(idv)
        surroundings, centers = Detector.get_srrds_centers(phenotype, problem.slt_space)
        probs = self.detector.get_probs(surroundings, centers)
        mid = (self.detector.theta1 + self.detector.theta2) / 2
        span = self.detector.theta2 - self.detector.theta1
        solution = idv[:]
        wrongs = len(np.where(probs <= self.detector.theta1)[0])
        uncertains = len(np.where(abs(probs - mid) < span / 2)[0])
        changes = len(np.where(solution != problem.original)[0])
        fitness = -(5 * wrongs + 3 * uncertains + changes)
        return fitness, wrongs, uncertains, changes

    def repair(self, level, max_epoch=20, time_budget=None, verbose=0):
        problem = self.build_problem(level)
        if problem is None:
            return level
        pop = [Individual.rand_initial(problem) for _ in range(self.pop_size)]
        self.repair_pop(pop, problem)
        for idv in pop:
            idv.fitness, *_ = self.evaluate(idv, problem)
        pop.sort(key=lambda x: x.fitness)

        best_fitness = pop[-1].fitness
        best = pop[-1]

        time_budget = time_budget if time_budget else float('inf')
        start_time = time.time()
        for t in range(max_epoch):
            offsprings = Repairer.cross_over(pop)
            self.mutation(offsprings, problem)
            self.repair_pop(offsprings, problem)
            best_wrongs = 100000
            for idv in offsprings:
                idv.fitness, wrongs, *_ = self.evaluate(idv, problem)
                if idv.fitness > best_fitness:
                    best_fitness = idv.fitness
                    best = idv
                    best_wrongs = wrongs
            if best_wrongs == 0:
                break
            pop = self.select(pop + offsprings)
            pop.sort(key=lambda x: x.fitness)

            if verbose:
                print(f'iteration {t+1}/{max_epoch}, best fitnees: {best_fitness}')
            if (time.time() - start_time) >= time_budget:
                break
            pass
        tmp = problem.decode(best)[1: -1, 1: -1]
        return MarioLevel.from_num_arr(tmp)

    @staticmethod
    def cross_over(pop):
        sample_scores = range(len(pop), 0, -1)
        total_score = sum(sample_scores)
        cumdist = list(itertools.accumulate(sample_scores))
        offsprings = []
        for _ in range(len(pop) // 2):
            p1 = pop[bisect.bisect(cumdist, random.random() * total_score)]
            p2 = pop[bisect.bisect(cumdist, random.random() * total_score)]
            s1 = p1[:].copy()
            s2 = p2[:].copy()
            x1, x2 = Individual(s1), Individual(s2)
            offsprings.append(x1)
            offsprings.append(x2)
        return offsprings

    def mutation(self, pop, problem):
        for idv in pop:
            indexes = []
            for i in range(len(idv.solution)):
                if random.random() > self.pm:
                    continue
                indexes.append(i)
            if not len(indexes):
                continue
            phenotype = problem.decode(idv)
            pos = problem.slt_space[indexes]
            surroundings, centers = Detector.get_srrds_centers(phenotype, pos)
            probs = self.detector.get_probs(surroundings, centers)
            for i, prob in zip(indexes, probs):
                if prob < self.detector.theta2:
                    idv.solution[i] = random.randrange(0, MarioLevel.num_tile_types)
        pass

    def repair_pop(self, pop, problem):
        for idv in pop:
            indexes = []
            for i in range(len(idv.solution)):
                if random.random() > self.pr:
                    continue
                indexes.append(i)
            if not len(indexes):
                continue
            phenotype = problem.decode(idv)
            pos = problem.slt_space[indexes]
            surroundings, centers = Detector.get_srrds_centers(phenotype, pos)
            predictions = self.detector.get_predictions(surroundings)
            probs = predictions[range(len(centers)), centers]

            for i, j, prob in zip(indexes, range(len(predictions)), probs):
                if prob < self.detector.theta1: # is wrong
                    candidates, = np.where(predictions[j] > 0.5)
                    if len(candidates):
                        idv.solution[i] = random.choice(candidates)
                    pass
                pass

    def build_problem(self, level):
        h, w = level.h + 2, level.w + 2
        padded_lvl = MarioLevel.num_tile_types * np.ones((h, w), dtype=np.int32)
        padded_lvl[1:-1, 1:-1] = level.to_num_arr()
        slt_space = self.detector.get_slt_space(padded_lvl)
        if not len(slt_space):
            return None
        return Problem(padded_lvl, slt_space)

    def select(self, pop):
        all_ids = [*range(len(pop))]
        selected_ids = []
        for _ in range(self.pop_size):
            tournaments = random.sample(all_ids, self.rrt)
            selected = max(tournaments, key=lambda x: pop[x].fitness)
            selected_ids.append(selected)
            all_ids.remove(selected)

        return [pop[i] for i in selected_ids]


class DivideConquerRepairer:
    def __init__(self, base_repairer=None, slicing_threshold=20, slice_time_budget=0.5):
        self.base_repairer = Repairer() if base_repairer is None else base_repairer
        self.slicing_threshold = slicing_threshold
        self.slice_time_budget = slice_time_budget

    def repair(self, level):
        slices = self.__divide(level)
        repaired = []
        for slc in slices:
            repaired.append(self.base_repairer.repair(slc))
        # print(repaired)
        return level_sum(repaired)

    def __divide(self, level):
        slices = []
        n_pipes = 0
        s, e = 0, 0
        ppset = MarioLevel.pipe_charset
        while e < level.w:
            col_pipes = len([c for c in level.content[:, e] if c in ppset])
            n_pipes += col_pipes
            e += 1
            if e == level.w or (n_pipes > self.slicing_threshold and col_pipes == 0):
                slices.append(level[:, s:e])
                s = e
                n_pipes = 0
        return slices


def check_pipes(lvl):
    lvl_num_arr = lvl.to_num_arr()
    h, w = lvl_num_arr.shape
    total = 0
    valid = 0
    for i, j in product(range(h), range(w)):
        tile = lvl_num_arr[i, j]
        if tile not in {6, 7, 8, 9}:
            continue
        total += 1
        if not 0 < i < h-1:
            continue
        above = lvl_num_arr[i-1, j]
        below = lvl_num_arr[i+1, j]
        if tile == 6 and j < w-1:
            right = lvl_num_arr[i, j+1]
            above_valid = above == 2
            below_valid = below == 8
            right_valid = right == 7
            valid += (above_valid and below_valid and right_valid)
        elif tile == 7 and j > 0:
            left = lvl_num_arr[i, j-1]
            above_valid = above in {2, 5}
            below_valid = below == 9
            left_valid = left == 6
            valid += (above_valid and below_valid and left_valid)
        elif tile == 8 and j < w-1:
            right = lvl_num_arr[i, j+1]
            above_valid = above in {6, 8}
            below_valid = below in {8, 0}
            right_valid = right == 9
            valid += (above_valid and below_valid and right_valid)
            pass
        elif tile == 9 and j > 0:
            left = lvl_num_arr[i, j-1]
            above_valid = above in {7, 9}
            below_valid = below in {9, 0}
            left_valid = left == 8
            valid += (above_valid and below_valid and left_valid)
    return valid, total
