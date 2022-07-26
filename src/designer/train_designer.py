"""
  @Time : 2021/11/29 20:03 
  @Author : Ziqi Wang
  @File : train_sac_designer.py 
"""
import json
import os
import time
import importlib
from src.algo.sac import SAC_Model, OffRewSAC_Trainer
from src.algo.ac_model import SquashedGaussianMLPActor, MLPQFunction
from src.environment.env import make_vec_offrew_env
from src.algo.replay_memory import ReplayMem
from src.gan.gan_config import nz
from src.utils.filesys import auto_dire, get_path
from src.environment.env_cfgs import history_len


def set_parser(parser):
    parser.add_argument(
        '--n_envs', type=int, default=5,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--eplen', type=int, default=25,
        help='Maximum nubmer of segments to generate in one episode'
    )
    parser.add_argument(
        '--total_steps', type=int, default=int(1e5),
        help='Total time steps (frames) for training SAC designer'
    )
    parser.add_argument('--gamma', type=float, default=0.7, help='Gamma parameter of RL')
    parser.add_argument('--tar_entropy', type=float, default=-nz, help='Target entropy of SAC')
    parser.add_argument('--tau', type=float, default=0.005, help='Tau parameter of SAC')
    parser.add_argument('--update_freq', type=int, default=10, help='Interval of updating SAC model')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size of training SAC model')
    parser.add_argument('--mem_size', type=int, default=int(5e5), help='Capacity of replay memory for SAC training')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for training the SAC model'
    )
    parser.add_argument(
        '--rfunc_name', type=str, default='default',
        help='Name of the file where the reward function located. '
             'The file must be put in the \'src.rfuncs\' package'
    )
    parser.add_argument(
        '--res_path', type=str, default='',
        help='Path related to \'/exp_data\' to save the training log. '
             'If not specified, a new folder named exp{id} will be created'
    )
    parser.add_argument(
        '--play_style', type=str, default='Runner',
        help='Play style (persona) of agent, \'Runner\', \'Killer\' and \'Collector\' are valid'
    )
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='Check points to save deisigner, specified by the number of time steps'
    )

def train_designer(cfgs):
    if not cfgs.res_path:
        res_path = auto_dire('exp_data')
    else:
        res_path = get_path('exp_data/' + cfgs.res_path)
        os.makedirs(res_path, exist_ok=True)

    rfunc = (
        importlib.import_module('src.environment.rfuncs')
        .__getattribute__(f'{cfgs.rfunc_name}')
    )
    with open(res_path + '/run_config.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write('---------SAC---------\n')
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(cfgs).items()
            if key not in {'rfunc_name', 'res_path', 'entry', 'check_points'}
        ]
        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))
    with open(f'{res_path}/kwargs.json', 'w') as f:
        json.dump({'hist_len': history_len}, f)

    env = make_vec_offrew_env(
        cfgs.n_envs, rfunc, res_path, cfgs.eplen, play_style=cfgs.play_style,
        log_itv=cfgs.n_envs * 2, device=cfgs.device, log_targets=['file', 'std']
    )

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    designer = SAC_Model(
        lambda: SquashedGaussianMLPActor(obs_dim, act_dim, [256, 256, 256]),
        lambda: MLPQFunction(obs_dim, act_dim, [256, 256, 256]),
        gamma=cfgs.gamma, tar_entropy=cfgs.tar_entropy, tau=cfgs.tau, device=cfgs.device
    )
    d_trainer = OffRewSAC_Trainer(
        env, cfgs.total_steps, cfgs.update_freq, cfgs.batch_size, ReplayMem(cfgs.mem_size),
        res_path, cfgs.check_points
    )
    d_trainer.train(designer)

