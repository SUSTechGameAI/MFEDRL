"""
@DATE: 2021/9/10
@Author: Ziqi Wang
@File: env.py
"""

import gym
import time
from torch import tensor
from typing import Optional, Callable, List
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from src.environment.env_info_logger import InfoCollector
from src.environment.reward_function import RewardFunc
from src.repairer.repairer import DivideConquerRepairer, Repairer
from src.utils.datastruct import RingQueue
from src.environment.env_cfgs import history_len
from src.gan.gan_use import *
from smb import *

#
# class GenerationEnv(gym.Env):
#     def __init__(self, rfunc=None, max_seg_num=100, archive_len=10):
#         self.rfunc = deepcopy(rfunc)
#         self.mario_proxy = MarioProxy()
#         self.action_space = gym.spaces.Box(-1, 1, (nz,))
#         self.observation_space = gym.spaces.Box(-1, 1, (archive_len * nz,))
#         self.simlt_buffer = RingQueue(2)
#         self.level_archive = RingQueue(archive_len)
#         self.latvec_archive = RingQueue(archive_len)
#         self.max_seg_num = max_seg_num
#         self.archive_len = archive_len
#         self.counter = 0
#         self.score = 0
#         self.repairer = Repairer()
#
#         self.onehot_seg = None
#         self.backup_latvec = None
#         self.backup_onehot_seg = None
#
#     def receive(self, **kwargs):
#         for key in kwargs.keys():
#             setattr(self, key, kwargs[key])
#
#     def step(self, action: np.ndarray):
#         seg = MarioLevel.from_one_hot_arr(self.onehot_seg)
#         self.latvec_archive.push(action)
#
#         self.counter += 1
#         self.simlt_buffer.push(seg)
#         if self.rfunc and self.rfunc.require_simlt:
#             simlt_res = self.__playable_test()
#         else:
#             simlt_res = None
#
#         archive = None if not len(self.level_archive) else self.level_archive.to_list()
#         reward = 0 if not self.rfunc else self.rfunc(archive=archive, seg=seg, simlt_res=simlt_res)
#
#         done = self.counter >= self.max_seg_num
#         self.score += reward
#
#         obs = self.__get_obs()
#         if done:
#             info = self.rfunc.reset()
#             info['TotalScore'] = self.score
#             info['EpLength'] = self.counter
#             # info['whole_level'] =
#         else:
#             info = {}
#         self.level_archive.push(seg)
#         return obs, reward, done, info
#
#     def __playable_test(self):
#         test_seg = level_sum(self.simlt_buffer.to_list())
#         return self.mario_proxy.simulate_game(test_seg)
#
#     def reset(self):
#         self.simlt_buffer.clear()
#         self.level_archive.clear()
#         self.latvec_archive.clear()
#         self.latvec_archive.push(self.backup_latvec)
#         self.level_archive.push(MarioLevel.from_one_hot_arr(self.backup_onehot_seg))
#         self.backup_latvec, self.backup_onehot_seg = None, None
#         self.score = 0
#         self.counter = 0
#         return self.__get_obs()
#
#     def __get_obs(self):
#         lack = self.archive_len - len(self.latvec_archive)
#         pad = [np.zeros([nz], np.float32) for _ in range(lack)]
#         return np.concatenate([*pad, *self.latvec_archive.to_list()])
#
#     def render(self, mode='human'):
#         pass


class OffRewardGenerationEnv(gym.Env):
    def __init__(self, rfunc=None, hist_len=history_len, eplen=25, return_lvl=False, init_one=False, play_style='Runner'):
        self.rfunc = RewardFunc() if rfunc is None else rfunc
        self.mario_proxy = MarioProxy() if self.rfunc.require_simlt else None
        self.action_space = gym.spaces.Box(-1, 1, (nz,))
        self.hist_len = hist_len
        self.observation_space = gym.spaces.Box(-1, 1, (hist_len * nz,))
        self.segs = []
        self.latvec_archive = RingQueue(hist_len)
        self.eplen = eplen
        self.counter = 0
        self.repairer = DivideConquerRepairer()
        self.init_one = init_one
        self.onehot_seg = None
        self.backup_latvec = None
        self.backup_onehot_seg = None
        self.return_lvl = return_lvl
        self.jagent = MarioJavaAgents.__getitem__(play_style)
        self.simlt_k = 4. if play_style == 'Runner' else 10.

    def receive(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def step(self, action: np.ndarray):
        seg = MarioLevel.from_one_hot_arr(self.onehot_seg)
        self.latvec_archive.push(action)

        self.counter += 1
        self.segs.append(seg)
        done = self.counter >= self.eplen
        if done:
            full_level = level_sum(self.segs)
            full_level = self.repairer.repair(full_level)
            w = MarioLevel.default_seg_width
            segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
            if self.mario_proxy:
                raw_simlt_res = self.mario_proxy.simulate_long(level_sum(segs), self.jagent, self.simlt_k)
                simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
            else:
                simlt_res = None
            rewards = self.rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            info = {}
            total_score = 0
            if self.return_lvl:
                info['LevelStr'] = str(full_level)
            for key in rewards:
                info[f'{key}_reward_list'] = rewards[key][-self.eplen:]
                info[f'{key}'] = sum(rewards[key][-self.eplen:])
                total_score += info[f'{key}']
            info['TotalScore'] = total_score
            info['EpLength'] = self.counter
        else:
            info = {}
        return self.__get_obs(), 0, done, info

    def reset(self):
        self.segs.clear()
        self.latvec_archive.clear()

        if self.init_one:
            self.latvec_archive.push(self.backup_latvec)
            self.segs.append(MarioLevel.from_one_hot_arr(self.backup_onehot_seg))
        else:
            for latvec, onehot_seg in zip(self.backup_latvec, self.backup_onehot_seg):
                self.latvec_archive.push(latvec)
                self.segs.append(MarioLevel.from_one_hot_arr(onehot_seg))
            # print(level_sum(self.segs))
        self.backup_latvec, self.backup_onehot_seg = None, None
        self.counter = 0
        return self.__get_obs()

    def __get_obs(self):
        lack = self.hist_len - len(self.latvec_archive)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        return np.concatenate([*pad, *self.latvec_archive.to_list()])

    def render(self, mode='human'):
        pass


class VecGenerationEnv(SubprocVecEnv):
    def __init__(
        self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None, hist_len=history_len,
        init_one=False, log_path=None, log_itv=-1, log_targets=None, device='cuda:0'
    ):
        super(VecGenerationEnv, self).__init__(env_fns, start_method)
        self.generator = get_generator(device=device)

        if log_path:
            self.logger = InfoCollector(log_path, log_itv, log_targets)
        else:
            self.logger = None
        self.hist_len = hist_len
        self.total_steps = 0
        self.start_time = time.time()
        self.device = device
        self.init_one = init_one

    def step_async(self, actions: np.ndarray) -> None:
        with torch.no_grad():
            z = torch.clamp(tensor(actions.astype(np.float32), device=self.device), -1, 1).view(-1, nz, 1, 1)
            onehot_segs = process_levels(self.generator(z))
        for remote, onehot_seg in zip(self.remotes, onehot_segs):
            remote.send(("env_method", ('receive', [], {'onehot_seg': onehot_seg})))
        for remote in self.remotes:
            remote.recv()
        for remote, action, onehot_seg in zip(self.remotes, actions, onehot_segs):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.total_steps += self.num_envs
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        envs_to_send = [i for i in range(self.num_envs) if dones[i]]
        self.send_reset_data(envs_to_send)

        if self.logger is not None:
            for i in range(self.num_envs):
                if infos[i]:
                    infos[i]['TotalSteps'] = self.total_steps
                    infos[i]['TimePassed'] = time.time() - self.start_time
            self.logger.on_step(dones, infos)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        self.send_reset_data()
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        self.send_reset_data()
        return _flatten_obs(obs, self.observation_space)

    def send_reset_data(self, env_ids=None):
        if env_ids is None:
            env_ids = [*range(self.num_envs)]
        target_remotes = self._get_target_remotes(env_ids)
        # latvecs = [
        #     np.random.rand(len(env_ids), nz).astype(np.float32) * 2 - 1
        #     for _ in range(history_len)
        # ]
        if self.init_one:
            latvecs = np.random.rand(len(env_ids), nz).astype(np.float32) * 2 - 1
        else:
            latvecs = np.random.rand(len(env_ids), self.hist_len, nz).astype(np.float32) * 2 - 1
        with torch.no_grad():
            if self.init_one:
                z = tensor(latvecs).view(-1, nz, 1, 1).to(self.device)
                onehot_segs = process_levels(self.generator(z))
            else:
                onehot_segs = [[] for _ in range(len(env_ids))]
                for i in range(len(env_ids)):
                    z = tensor(latvecs[i]).view(-1, nz, 1, 1).to(self.device)
                    onehot_segs[i] = process_levels(self.generator(z))
        for remote, latvec, onehot_seg in zip(target_remotes, latvecs, onehot_segs):
            kwargs = {'backup_latvec': latvec, 'backup_onehot_seg': onehot_seg}
            remote.send(("env_method", ('receive', [], kwargs)))
        for remote in target_remotes:
            remote.recv()

    def close(self) -> None:
        super().close()
        if self.logger is not None:
            self.logger.close()


def make_vec_offrew_env(
        num_envs, rfunc=None, log_path=None, eplen=25, log_itv=-1, hist_len=history_len, init_one=False,
        play_style='Runner', device='cuda:0', log_targets=None, return_lvl=False
    ):
    return make_vec_env(
        OffRewardGenerationEnv, n_envs=num_envs, vec_env_cls=VecGenerationEnv,
        vec_env_kwargs={
            'log_path': log_path,
            'log_itv': log_itv,
            'log_targets': log_targets,
            'device': device,
            'hist_len': hist_len,
            'init_one': init_one
        },
        env_kwargs={
            'rfunc': rfunc,
            'eplen': eplen,
            'return_lvl': return_lvl,
            'play_style': play_style,
            'hist_len': hist_len,
            'init_one': init_one
        }
    )

# def make_vec_generation_env(
#         num_envs, rfunc=None, log_path=None, max_seg_num=100, archive_len=8,
#         log_itv=-1, device='cuda:0', log_targets=None
#     ):
#     return make_vec_env(
#         GenerationEnv, n_envs=num_envs, vec_env_cls=VecGenerationEnv,
#         vec_env_kwargs={
#             'log_path': log_path,
#             'log_itv': log_itv,
#             'log_targets': log_targets,
#             'device': device
#         },
#         env_kwargs={
#             'rfunc': rfunc,
#             'max_seg_num': max_seg_num,
#             'archive_len': archive_len,
#         }
#     )
#
