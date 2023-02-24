"""
  @Time : 2021/9/27 20:25 
  @Author : Ziqi Wang
  @File : env_info_logger.py
"""

import json
import numpy as np


class InfoCollector:
    ignored_keys = {'episode', 'terminal_observation'}
    save_itv = 1000

    def __init__(self, path, log_itv=100, log_targets=None):
        self.data = []
        self.path = path
        self.msg_itv = log_itv
        self.time_before_save = InfoCollector.save_itv
        self.msg_ptr = 0
        self.log_targets = [] if log_targets is None else log_targets
        if 'file' in log_targets:
            with open(f'{self.path}/mylog.txt', 'w') as f:
                f.write('')
        self.recent_time = 0

    def on_step(self, dones, infos):
        for done, info in zip(dones, infos):
            if done:
                self.data.append({
                    key: val for key, val in info.items()
                    if key not in InfoCollector.ignored_keys and 'reward_list' not in key
                })
                self.time_before_save -= 1
        if self.time_before_save <= 0:
            with open(f'{self.path}/ep_infos.json', 'w') as f:
                json.dump(self.data, f)
            self.time_before_save += InfoCollector.save_itv

        if self.log_targets and 0 < self.msg_itv <= (len(self.data) - self.msg_ptr):
            keys = set(self.data[-1].keys()) - {'TotalSteps', 'TimePassed', 'TotalScore', 'EpLen'}

            msg = '%sTotal steps: %d%s\n' % ('-' * 16, self.data[-1]['TotalSteps'], '-' * 16)
            msg += 'Time passed: %ds\n' % self.data[-1]['TimePassed']
            t = self.data[-1]['TimePassed'] - self.recent_time
            self.recent_time = self.data[-1]['TimePassed']
            f = sum(item['EpLength'] for item in self.data[self.msg_ptr:])
            msg += 'fps: %.3g\n' % (f/t)
            for key in keys:
                values = [item[key] for item in self.data[self.msg_ptr:]]
                values = np.array(values)
                msg += '%s: %.2f +- %.2f\n' % (key, values.mean(), values.std())
            values = [item['TotalScore'] for item in self.data[self.msg_ptr:]]
            values = np.array(values)
            msg += 'TotalScore: %.2f +- %.2f\n' % (values.mean(), values.std())

            if 'file' in self.log_targets:
                with open(f'{self.path}/mylog.txt', 'ztraces') as f:
                    f.write(msg + '\n')
            if 'std' in self.log_targets:
                print(msg)
            self.msg_ptr = len(self.data)
            pass

    def close(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)
