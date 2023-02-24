"""
  @Time : 2021/9/8 17:05 
  @Author : Ziqi Wang
  @File : smb.py
"""
import glob
import re

import jpype
import numpy as np
import pygame as pg
from math import ceil
from enum import Enum
from root import PRJROOT
from jpype import JString
from config import JVMPath
from typing import Union, List, Dict
from src.utils.filesys import get_path
from itertools import product, accumulate

# ! ! This file must be placed at the project root directory ! !


class MarioLevel:
    tex_size = 16
    height = 14
    default_seg_width = 28
    mapping = {
        'i-c': ('X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']', 'o'),
        'c-i': {'X': 0, 'S': 1, '-': 2, '?': 3, 'Q': 4, 'E': 5, '<': 6,
        '>': 7, '[': 8, ']': 9, 'o': 10}
    }
    empty_tiles = {'-', 'E', 'o'}
    num_tile_types = len(mapping['i-c'])
    pipe_charset = {'<', '>', '[', ']'}
    pipe_intset = {6, 7, 8, 9}
    textures = [
        pg.image.load(PRJROOT + f'assets/tile-{i}.png')
        for i in range(num_tile_types)
    ]

    def __init__(self, content):
        if isinstance(content, np.ndarray):
            self.content = content
        else:
            tmp = [list(line) for line in content.split('\n')]
            while not tmp[-1]:
                tmp.pop()
            self.content = np.array(tmp)
        self.h, self.w = self.content.shape
        self.__tile_pttr_cnts = {}
        self.attr_dict = {}

    def to_num_arr(self):
        res = np.zeros((self.h, self.w), int)
        for i, j in product(range(self.h), range(self.w)):
            char = self.content[i, j]
            res[i, j] = MarioLevel.mapping['c-i'][char]
        return res

    def to_img(self, save_path='render.png') -> pg.Surface:
        tex_size = MarioLevel.tex_size
        num_lvl = self.to_num_arr()
        # img = Image.new('RGBA', (self.w * tex_size, self.h * tex_size), (80, 80, 255, 255))
        img = pg.Surface((self.w * tex_size, self.h * tex_size))
        img.fill((150, 150, 255))

        for i, j in product(range(self.h), range(self.w)):
            tile_id = num_lvl[i, j]
            if tile_id == 2:
                continue
            img.blit(
                MarioLevel.textures[tile_id],
                (j * tex_size, i * tex_size, tex_size, tex_size)
            )
        if save_path:
            safe_path = get_path(save_path)
            pg.image.save(img, safe_path)
        return img

    def to_img_with_trace(self, trace, save_path='render_with_trace.png', lw=3):
        img = self.to_img('')
        p = 0
        while p < len(trace) and trace[p][0] < self.w * MarioLevel.tex_size:
            p += 1
        pg.draw.lines(img, 'black', False, [(x, y-8) for x, y in trace[:p]], lw)
        if save_path is not None:
            pg.image.save(img, get_path(save_path))
        return img

    def save(self, fpath):
        safe_path = get_path(fpath + '.smblv')
        with open(safe_path, 'w') as f:
            f.write(str(self))

    def tile_pattern_counts(self, w=2):
        if not w in self.__tile_pttr_cnts.keys():
            counts = {}
            for i, j in product(range(self.h - w + 1), range(self.w - w + 1)):
                key = ''.join(self.content[i+x][j+y] for x, y in product(range(w), range(w)))
                count = counts.setdefault(key, 0)
                counts[key] = count + 1
            self.__tile_pttr_cnts[w] = counts
        return self.__tile_pttr_cnts[w]

    def tile_pattern_distribution(self, w=2):
        counts = self.tile_pattern_counts(w)
        C = (self.h - w + 1) * (self.w - w + 1)
        return {key: val / C for key, val in counts.items()}

    def to_segs(self):
        W = MarioLevel.default_seg_width
        return [self[:, s:s+W] for s in range(0, self.w, W)]

    def __getattr__(self, item):
        if item == 'shape':
            return self.content.shape
        elif item == 'h':
            return self.content.shape[0]
        elif item == 'w':
            return self.content.shape[1]
        elif item not in self.attr_dict.keys():
            if item == 'n_gaps':
                empty_map1 = np.where(self.content[-1] in MarioLevel.empty_tiles, 1, 0)
                empty_map2 = np.where(self.content[-2] in MarioLevel.empty_tiles, 1, 0)
                res = len(np.where(empty_map1 + empty_map2 == 2))
                self.attr_dict['n_ground'] = res
            elif item == 'n_enemies':
                self.attr_dict['n_enemies'] = str(self).count('E')
            elif item == 'n_coins':
                self.attr_dict['n_coins'] = str(self).count('o')
            elif item == 'n_questions':
                self.attr_dict['n_questions'] = str(self).count('Q')
            elif item == 'n_empties':
                empty_map = np.where(self.content in MarioLevel.empty_tiles)
                self.attr_dict['n_questions'] = len(empty_map)
        return self.attr_dict[item]

    def __str__(self):
        lines = [''.join(line) + '\n' for line in self.content]
        return ''.join(lines)

    def __add__(self, other):
        concated = np.concatenate([self.content, other.content], axis=1)
        return MarioLevel(concated)

    def __getitem__(self, item):
        return MarioLevel(self.content[item])

    @staticmethod
    def from_num_arr(num_arr):
        h, w = num_arr.shape
        res = np.empty((h, w), str)
        for i, j in product(range(h), range(w)):
            tile_id = num_arr[i, j]
            res[i, j] = MarioLevel.mapping['i-c'][int(tile_id)]
        return MarioLevel(res)

    @staticmethod
    def from_file(fpath):
        safe_path = get_path(fpath)
        with open(safe_path, 'r') as f:
            return MarioLevel(f.read())

    @staticmethod
    def from_one_hot_arr(one_hot_arr: np.ndarray):
        num_lvl = one_hot_arr.argmax(axis=0)
        return MarioLevel.from_num_arr(num_lvl)


class MarioJavaAgents(Enum):
    Runner = 'agents.robinBaumgarten'
    Killer = 'agents.killer'
    Collector = 'agents.collector'

    def __str__(self):
        return self.value + '.Agent'


class MarioProxy:
    # __jmario = jpype.JClass("MarioProxy")()

    def __init__(self):
        if not jpype.isJVMStarted():
            jpype.startJVM(
                jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
                f"-Djava.class.path={PRJROOT}Mario-AI-Framework.jar", '-Xmx2g'
            )
            """
                -Xmx{size} set the heap size.
            """
        self.__proxy = jpype.JClass("MarioProxy")()

    @staticmethod
    def __extract_res(jresult):
        return {
            'status': str(jresult.getGameStatus().toString()),
            'completing-ratio': float(jresult.getCompletionPercentage()),
            '#kills': int(jresult.getKillsTotal()),
            '#kills-by-fire': int(jresult.getKillsByFire()),
            '#kills-by-stomp': int(jresult.getKillsByStomp()),
            '#kills-by-shell': int(jresult.getKillsByShell()),
            'trace': [
                [float(item.getMarioX()), float(item.getMarioY())]
                for item in jresult.getAgentEvents()
            ]
        }

    def play_game(self, level: Union[str, MarioLevel]):
        if type(level) == str:
            jfilepath = JString(level)
            jresult = self.__proxy.playGameFromTxt(jfilepath)
        else:
            jresult = self.__proxy.playGame(JString(str(level)))
        return MarioProxy.__extract_res(jresult)

    def simulate_game(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Runner,
        render: bool=False
    ) -> Dict:
        """
        Run simulation with an agent for ztraces given level
        :param level: if type is str, must be path of ztraces valid level file.
        :param agent: type of the agent.
        :param render: render or not.
        :return: dictionary of the results.
        """
        # start_time = time.perf_counter()
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_file(level)
        # real_time_limit_ms = 2 * (level.w * 15 + 1000)
        real_time_limit_ms = level.w * 115 + 1000 if not render else 200000
        jresult = self.__proxy.simulateGame(JString(str(level)), jagent, render, real_time_limit_ms, 30)
        # Refer to Mario-AI-Framework.engine.core.MarioResult, add more entries if need be.
        return MarioProxy.__extract_res(jresult)

    def simulate_long(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Runner,
        k: float=4., b: int=100
    ) -> Dict:
        # start_time = time.perf_counter()
        ts = MarioLevel.tex_size
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_file(level)
        reached_tile = 0
        res = {'restarts': [], 'trace': []}
        dx = 0
        while reached_tile < level.w - 1:
            jresult = self.__proxy.simulateWithRealTimeSuspension(JString(str(level[:, reached_tile:])), jagent, k, b)
            pyresult = MarioProxy.__extract_res(jresult)

            reached = pyresult['trace'][-1][0]
            reached_tile += ceil(reached / ts)
            if pyresult['status'] != 'WIN':
                res['restarts'].append(reached_tile)
            res['trace'] += [[dx + item[0], item[1]] for item in pyresult['trace']]
            dx = reached_tile * ts
        return res

    @staticmethod
    def get_seg_infos(full_info, check_points=None):
        restarts, trace = full_info['restarts'], full_info['trace']
        W = MarioLevel.default_seg_width
        ts = MarioLevel.tex_size
        if check_points is None:
            end = ceil(trace[-1][0] / ts)
            check_points = [x for x in range(W, end, W)]
            check_points.append(end)
        res = [{'trace': [], 'playable': True} for _ in check_points]
        s, e, i = 0, 0, 0
        restart_pointer = 0
        # dx = 0
        while True:
            while e < len(trace) and trace[e][0] < ts * check_points[i]:
                if restart_pointer < len(restarts) and restarts[restart_pointer] < check_points[i]:
                    res[i]['playable'] = False
                    restart_pointer += 1
                e += 1
            x0 = trace[s][0]
            res[i]['trace'] = [[item[0] - x0, item[1]] for item in trace[s:e]]
            # x0, y0 = trace[s]
            # res[i]['trace'] = [[item[0] - x0, item[1] - y0] for item in trace[s:e]]
            # dx = ts * check_points[i]
            i += 1
            if i == len(check_points):
                break
            s = e
        return res


def save_batch(lvls, fname):
    contents = [str(lvl).strip() for lvl in lvls]
    content = '\n;\n'.join(contents)
    if len(fname) <= 5 or fname[-5:] != '.smblvs':
        fname += '.smblvs'
    with open(get_path(fname), 'w') as f:
        f.write(content)
    pass

def load_batch(fname):
    with open(get_path(fname), 'r') as f:
        content = f.read()
    return [MarioLevel(c) for c in content.split('\n;\n')]

def level_sum(lvls) -> MarioLevel:
    if type(lvls[0]) == MarioLevel:
        concated_content = np.concatenate([l.content for l in lvls], axis=1)
    else:
        concated_content = np.concatenate([l for l in lvls], axis=1)
    return MarioLevel(concated_content)

def traverse_level_files(path='levels/train'):
    for lvl_path in glob.glob(get_path(f'{path}\\*.smblv')):
        lvl = MarioLevel.from_file(lvl_path)
        name = re.split('[/\\\\]', lvl_path)[-1][:-6]
        yield lvl, name

def traverse_batched_level_files(path):
    for lvl_path in glob.glob(get_path(f'{path}\\*.smblvs')):
        name = re.split('[/\\\\]', lvl_path)[-1][:-7]
        with open(lvl_path, 'r') as f:
            txt = f.read()
        lvls = []
        for lvlstr in txt.split(';\n'):
            if len(lvlstr) < 10:
                continue
            lvls.append(MarioLevel(lvlstr))
        yield lvls, name


if __name__ == '__main__':
    lvl = MarioLevel.from_file('levels/original/mario-1-1.smblv')
    MarioProxy().play_game(lvl)
    pass

