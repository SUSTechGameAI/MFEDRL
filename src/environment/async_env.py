"""
  @Time : 2022/3/18 18:13 
  @Author : Ziqi Wang
  @File : async_env.py
"""

import time

import jpype
import torch
import numpy as np
from smb import level_sum, MarioProxy
from src.environment.reward_function import RewardFunc, FunContent, FunBehaviour, Playability
from src.utils.datastruct import RingQueue
from src.utils.parallel import AsyncPoolWithLoad, TaskLoad
from src.repairer.repairer import DivideConquerRepairer
from src.gan.gan_use import *
from src.environment.env_cfgs import history_len


rfunc = RewardFunc(FunContent(), FunBehaviour(), Playability())


class AsyncGenerationEnv:
    repairer = DivideConquerRepairer()

    def __init__(self, rew_parallel=5, l=50, g_device='cuda:0', logger=None):
        self.rew_prl = rew_parallel
        self.reward_pool = (
            None if rfunc is None else
            AsyncPoolWithLoad(rew_parallel)
        )
        self.eplen = l
        self.hist_len = history_len
        self.generator = get_generator(device=g_device)
        self.logger = AsyncGenerationEnvLogger() if logger is None else logger
        self.logger.eplen = l
        self.g_device = g_device
        self.pid = 0
        self.counter = 0
        # self.total_steps = 0
        self.obs_buffer = RingQueue(history_len)
        self.latvecs = []
        self.a_buffer = {}
        self.o_buffer = {}
        self.op_buffer = {}

    def step(self, action):
        # self.total_steps += 1
        self.counter += 1
        tmp = action.squeeze()
        self.obs_buffer.push(tmp)
        self.latvecs.append(tmp)
        obs = self.__get_obs()
        done = self.counter == self.eplen
        if not rfunc is None:
            self.a_buffer[self.pid].append(tmp)
            self.op_buffer[self.pid].append(obs)
            if not done:
                self.o_buffer[self.pid].append(obs)
        if done:
            self.__on_terminate()
            self.logger.on_episode()
        return obs, done

    def __on_terminate(self):
        z = torch.tensor(np.array(self.latvecs), device=self.g_device).view(-1, nz, 1, 1)
        str_segs = [str(item) for item in process_levels(self.generator(z), True)]
        if not rfunc is None:
            self.reward_pool.push(AsyncGenerationEnv.reward, TaskLoad(str_segs, pid=self.pid))
            self.pid += 1

    def reset(self):
        self.counter = 0
        self.latvecs.clear()
        # for _ in range(self.hist_len):
        z0 = np.random.rand(nz).astype(np.float32) * 2 - 1
        self.latvecs.append(z0)
        self.obs_buffer.push(z0)
        obs = self.__get_obs()
        if not rfunc is None:
            self.o_buffer[self.pid] = [obs]
            self.a_buffer[self.pid] = []
            self.op_buffer[self.pid] = []
        return obs

    def __get_obs(self):
        # if len(self.latvecs) >= history_len:
        #     return np.concatenate(self.latvecs[-self.hist_len:])
        lack = history_len - len(self.obs_buffer)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        return np.concatenate([*pad, *self.obs_buffer.to_list()])

    def roll_out(self):
        """
        If no reward function is set, don't call this method
        :return: [<o,a,r,op,d>]
        """
        reward_res = self.reward_pool.collect()
        while self.reward_pool.get_num_waiting() > 2 * self.rew_prl:
            reward_res += self.reward_pool.collect()
            time.sleep(1)
        self.logger.on_roll_out(reward_res)
        return self.__match_trajs(reward_res)

    def close(self):
        reward_res = self.reward_pool.wait_and_get()
        return self.__match_trajs(reward_res)

    def __match_trajs(self, reward_res):
        if rfunc is None:
            return None
        transistions = []
        for rewards, info in reward_res:
            pid = info['pid']
            keys = rewards.keys()
            transistions += [(
                self.o_buffer[pid][i],
                self.a_buffer[pid][i],
                sum(rewards[key][i] for key in keys),
                self.op_buffer[pid][i],
                (i == self.eplen - 1)
                ) for i in range(self.eplen)
            ]
            # print(*(item[2] for item in transistions))
            # print(*transistions[-1], sep='\n')
            # print('-' * 40)
            self.o_buffer.pop(pid)
            self.a_buffer.pop(pid)
            self.op_buffer.pop(pid)
        return transistions

    @staticmethod
    def reward(str_segs):
        simulator = MarioProxy()
        full_lvl = level_sum(list(MarioLevel(seg) for seg in str_segs))
        full_lvl = AsyncGenerationEnv.repairer.repair(full_lvl)
        tmp = simulator.simulate_long(full_lvl)
        simlt_res = MarioProxy.get_seg_infos(tmp)
        W = MarioLevel.default_seg_width
        N = full_lvl.w // W
        res = rfunc.get_rewards(
            segs=[full_lvl[:, i * W:(i + 1) * W] for i in range(N)],
            simlt_res=simlt_res
        )
        return res


class AsyncGenerationEnvLogger:
    def __init__(self, log_itv=100, path=None, std=True):
        self.buffer = []
        self.path = path
        self.std = std
        self.log_itv = log_itv
        self.path = path
        if not path is None:
            self.path = get_path(f'{path}/mylog.txt')
            with open(self.path, 'w') as f:
                f.write('')
        self.eplen = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.ep = 0
        self.updates = 0

    def on_model_update(self):
        self.updates += 1

    def on_roll_out(self, reward_res):
        for rewards, _ in reward_res:
            self.buffer.append({key: sum(vals) for key, vals in rewards.items()})

    def on_episode(self):
        self.ep += 1
        if self.ep % self.log_itv == 0:
            msg = '%sTotal steps: %d%s\n' % ('-' * 12, self.ep * self.eplen, '-' * 12)
            msg += 'Time passed: %ds\n' % (time.time() - self.start_time)
            msg += 'Fps: %.3g\n' % (self.eplen * self.log_itv / (time.time() - self.last_time))
            msg += 'Model updates: %d' % self.updates
            if len(self.buffer):
                msg += '\n'
                total_rewards = 0
                for key in self.buffer[-1].keys():
                    tmp = np.array([item[key] for item in self.buffer])
                    msg += '%s: %.4g +- %.3g\n' % (key, tmp.mean(), tmp.std())
                    total_rewards = total_rewards + tmp
                msg += 'TotalScore: %.4g +- %.3g' % (total_rewards.mean(), total_rewards.std())

            if self.std:
                print(msg)
            if not self.path is None:
                with open(self.path, 'a') as f:
                    f.write(msg + '\n')

            self.buffer.clear()
            self.last_time = time.time()

    # def close(self):
    #     if not self.f is None:
    #         self.f.close()

