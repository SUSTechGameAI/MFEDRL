"""
  @Time : 2021/11/29 19:56 
  @Author : Ziqi Wang
  @File : sac.py 
"""

import os
import time

import torch
import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from src.algo.replay_memory import ReplayMem
from src.gan.gan_config import nz
from src.utils.filesys import get_path


class SAC_Model:
    def __init__(self, netA_builder, netQ_builder, gamma=0.99, tau=0.005, tar_entropy=-nz, device='cuda:0'):
        self.netA = netA_builder().to(device)
        self.netQ1 = netQ_builder().to(device)
        self.netQ2 = netQ_builder().to(device)
        self.netA_optimizer = torch.optim.Adam(self.netA.parameters(), 3e-4)
        self.netQ1_optimizer = torch.optim.Adam(self.netQ1.parameters(), 3e-4)
        self.netQ2_optimizer = torch.optim.Adam(self.netQ2.parameters(), 3e-4)

        self.tar_netQ1 = netQ_builder().to(device)
        self.tar_netQ2 = netQ_builder().to(device)
        self.tar_netQ1.load_state_dict(self.netQ1.state_dict())
        self.tar_netQ2.load_state_dict(self.netQ2.state_dict())
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.log_alpha = torch.tensor([1], dtype=torch.float, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], 3e-4)
        self.tar_entropy = torch.tensor([tar_entropy], device=device, requires_grad=False)

    def make_decision(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a, _ = self.netA(torch.tensor(obs, device=self.device), with_logprob=False)
            a = a.to('cpu').numpy()
        return a.astype(np.float32)

    def update(self, batch):
        s, a, _, _, _ = batch
        y = self.process_batch(batch)
        self.update_critic(s, a, y)
        self.update_actor(s)
        self.update_alpha(s)
        self.update_tar_nets()

    def process_batch(self, batch):
        # s, ztraces, r, sp, d = batch
        s, _, r, sp, _ = batch
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            ap, log_ap = self.netA(sp)
            tar_q1 = self.tar_netQ1(sp, ap)
            tar_q2 = self.tar_netQ2(sp, ap)
            tar_q = torch.min(tar_q1, tar_q2).squeeze()
            # The terminate is fake, thus no (1-d) is multiplied
            y = r + self.gamma * (tar_q - alpha * log_ap)
        return y.float().unsqueeze(-1)

    def update_critic(self, s_batch, a_batch, y):
        self.netQ1_optimizer.zero_grad()
        self.netQ2_optimizer.zero_grad()
        q1_loss = F.mse_loss(self.netQ1(s_batch, a_batch), y)
        q2_loss = F.mse_loss(self.netQ2(s_batch, a_batch), y)
        q1_loss.backward()
        q2_loss.backward()
        self.netQ1_optimizer.step()
        self.netQ2_optimizer.step()

    def update_actor(self, s_batch):
        alpha = torch.exp(self.log_alpha)
        a, log_a = self.netA(s_batch)
        value_a = torch.min(self.netQ1(s_batch, a), self.netQ2(s_batch, a))
        self.netA_optimizer.zero_grad()
        a_loss = (alpha * log_a - value_a).mean()
        a_loss.backward()
        self.netA_optimizer.step()
        pass

    def update_alpha(self, s_batch):
        with torch.no_grad():
            self.alpha_optimizer.zero_grad()
        a, log_a = self.netA(s_batch)
        loss_alpha = -(self.log_alpha * (log_a + self.tar_entropy).detach()).mean()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        pass

    def update_tar_nets(self):
        polyak_update(self.netQ1.parameters(), self.tar_netQ1.parameters(), self.tau)
        polyak_update(self.netQ2.parameters(), self.tar_netQ2.parameters(), self.tau)

    def save(self, path, fmt='%s', only_actor=True):
        torch.save(self.netA, path + '/' + fmt % 'actor' + '.pth')
        if not only_actor:
            torch.save(self.netQ1, path + '/' + fmt % 'critic1' + '.pth')
            torch.save(self.netQ2, path + '/' + fmt % 'critic2' + '.pth')
            torch.save(self.tar_netQ1, path + '/' + fmt % 'tar_critic1' + '.pth')
            torch.save(self.tar_netQ2, path + '/' + fmt % 'tar_critic2' + '.pth')


class AsyncGenerativeSAC_Trainer:
    def __init__(self, update_itv=5, update_ratio=5, batch_size=256, rep_mem=None):
        # self.update_freq=update_freq
        self.update_itv = update_itv
        self.update_ratio = update_ratio
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        pass

    def train(self, model, env, step_budget, save_path='./', check_points=None):
        steps =  0
        update_counter = 0
        # self.rep_mem.clear()

        miss_counter = 0
        start_time = time.time()
        ep = 0
        print('Start to train AGSAC')
        while steps < step_budget:
            ep += 1
            done = False
            obs = env.reset()
            while not done:
                action = model.make_decision(np.expand_dims(obs, 0))
                next_ob, done = env.step(action)
                update_counter += 1
                steps += 1
            transitions = env.roll_out()
            for t in transitions:
                self.rep_mem.add(*t)
            # self.rep_mem.add_trajectories(*transitions)
            if len(self.rep_mem) < self.batch_size:
                miss_counter += update_counter
                update_counter = 0
            elif ep % self.update_itv == 0:
                repeats = update_counter // self.update_ratio
                for _ in range(repeats):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)
                    env.logger.on_model_update()
                update_counter %= self.update_ratio

            if check_points and steps >= check_points[0]:
                check_point_path = get_path(save_path + f'/model_at_{steps}')
                os.makedirs(check_point_path, exist_ok=True)
                model.save(check_point_path)
                check_points.pop(0)

        transitions = env.close()
        for t in transitions:
            self.rep_mem.add(*t)
        repeats = miss_counter // self.update_ratio
        print(f'Update model for {repeats} times compensatorily')
        for _ in range(repeats):
            batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
            model.update(batch_data)
        print(f'Training finished in {time.time()-start_time}s')
        model.save(save_path)
        pass


class OffRewSAC_Trainer:
    def __init__(self, env, step_budget, update_freq=10, batch_size=384, rep_mem=None, save_path = '.', check_points = None):
        self.env = env
        self.n_parallel = env.num_envs
        self.step_budget = step_budget
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.steps = 0
        self.check_points = [] if not check_points else check_points
        self.check_points.sort(reverse=True)
        self.save_path = save_path
        pass

    def train(self, model):
        self.steps = 0

        obs = self.env.reset()
        print('Start to train SAC')
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        action_buffer = [[] for _ in range(self.env.num_envs)]
        next_obs_buffer = [[] for _ in range(self.env.num_envs)]
        new_transitions = 0
        while self.steps < self.step_budget:
            actions = model.make_decision(obs)
            next_obs, _, dones, infos = self.env.step(actions)
            for i in range(len(next_obs)):
                if dones[i]:
                    next_obs[i] = infos[i]['terminal_observation']
            for i, (ob, action, next_ob) in enumerate(zip(obs, actions, next_obs)):
                obs_buffer[i].append(ob)
                action_buffer[i].append(action)
                next_obs_buffer[i].append(next_ob)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            for i, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                reward_lists = []
                for key in info.keys():
                    if 'reward_list' not in key:
                        continue
                    reward_lists.append(info[key])
                rewards = []

                for j in range(len(reward_lists[0])):
                    step_reward = 0
                    for item in reward_lists:
                        step_reward += item[j]
                    rewards.append(step_reward)
                self.rep_mem.add_batched(
                    obs_buffer[i], action_buffer[i], rewards, next_obs_buffer[i],
                    [False] * (len(reward_lists[0]) - 1) + [True]
                )
                obs_buffer[i].clear()
                action_buffer[i].clear()
                next_obs_buffer[i].clear()

                new_transitions += len(reward_lists[0])
            if new_transitions > self.update_freq and len(self.rep_mem) > self.batch_size:
                update_times = new_transitions // self.update_freq
                for _ in range(update_times):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)
                new_transitions = new_transitions % self.update_freq

            if len(self.check_points) and self.steps >= self.check_points[-1]:
                check_point_path = get_path(self.save_path + f'/model_at_{self.steps}')
                os.makedirs(check_point_path, exist_ok=True)
                model.save(check_point_path)
                self.check_points.pop()
                pass

        model.save(self.save_path)
        pass


class SAC_Trainer:
    def __init__(
        self, env, step_budget, update_itv=500, update_repeats=100, update_start=None, batch_size=256,
        rep_mem=None, save_path = '.', check_points = None
    ):
        self.env = env
        self.n_parallel = env.num_envs
        self.step_budget = step_budget
        self.update_itv  = update_itv
        self.update_start = batch_size if update_start is None else update_start
        self.update_repeats = update_repeats
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.steps = 0
        self.check_points = [] if not check_points else check_points
        self.check_points.sort(reverse=True)
        self.save_path = save_path
        pass

    def train(self, model):
        self.steps = 0
        update_wait = self.update_start
        obs = self.env.reset()
        print('Start to train SAC')
        while self.steps < self.step_budget:
            actions = model.make_decision(obs)
            next_obs, rewards, dones, infos = self.env.step(actions)
            self.rep_mem.add_batched(obs, actions, rewards, next_obs, dones)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            update_wait -= self.n_parallel
            if update_wait <= 0:
                for _ in range(self.update_repeats):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)
                update_wait = self.update_itv
            if len(self.check_points) and self.steps >= self.check_points[-1]:
                check_point_path = self.save_path + f'/model_at_{self.steps}'
                os.makedirs(check_point_path, exist_ok=True)
                model.save(check_point_path)
                self.check_points.pop()
                pass

        model.save(self.save_path)
        pass

