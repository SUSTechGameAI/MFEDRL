"""
  @Time : 2022/3/20 20:19 
  @Author : Ziqi Wang
  @File : train_agsac_designer.py 
"""
import json

"""
  @Time : 2021/11/29 20:03 
  @Author : Ziqi Wang
  @File : train_sac_designer.py 
"""

import os
import time
import importlib
from root import PRJROOT
from src.algo.sac import SAC_Model, AsyncGenerativeSAC_Trainer
from src.algo.ac_model import SquashedGaussianMLPActor, MLPQFunction
from src.environment import async_env
from src.algo.replay_memory import ReplayMem
from src.utils.filesys import auto_dire, get_path
from src.gan.gan_config import nz
from src.environment.env_cfgs import history_len


def set_parser(parser):
    parser.add_argument(
        '--n_envs', type=int, default=5,
        help='Number of parallel environments.'
    )
    parser.add_argument(
        '--max_seg_num', type=int, default=50,
        help='Maximum nubmer of segments to generate in the generation enviroment.'
    )
    # parser.add_argument(
    #     '--hist_len', type=int, default=7,
    #     help='Maximum nubmer of segments to generate in the generation enviroment.'
    # )
    parser.add_argument(
        '--total_steps', type=int, default=int(1e5),
        help='Total time steps (frames) for training PPO designer.'
    )
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--tar_entropy', type=float, default=-nz)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--update_itv', type=int, default=50)
    parser.add_argument('--update_ratio', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=384)
    parser.add_argument('--mem_size', type=int, default=int(5e5))
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for training the AGSAC agent.'
    )
    parser.add_argument(
        '--rfunc_name', type=str, default='default',
        help='Name of the file where the reward function located. '
             'The file must be put in the \'src.reward_functions\' package.'
    )
    parser.add_argument(
        '--res_path', type=str, default='',
        help='Path relateed to \'/exp_data\'to save the training log. '
             'If not specified, a new folder named exp{id} will be created.'
    )
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='check points to save deisigner, specified by the number of time steps.'
    )

def train_designer(cfgs):
    if cfgs.res_path == '':
        res_path = auto_dire(PRJROOT + 'exp_data')
    else:
        res_path = get_path(f'exp_data/{cfgs.res_path}')
        os.makedirs(res_path, exist_ok=True)

    rfunc = (
        importlib.import_module('src.environment.rfuncs')
        .__getattribute__(f'{cfgs.rfunc_name}')
    )
    with open(res_path + '/run_config.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write('---------AGSAC---------\n')
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(cfgs).items()
            if key not in {'rfunc_name', 'res_path', 'entry', 'check_points'}
        ]
        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))

    with open(f'{res_path}/kwargs.json', 'w') as f:
        json.dump({'hist_len': history_len}, f)

    async_env.AsyncGenerationEnv.rfunc = rfunc
    logger = async_env.AsyncGenerationEnvLogger(path=res_path, log_itv=cfgs.n_envs * 2)
    env = async_env.AsyncGenerationEnv(cfgs.n_envs, cfgs.max_seg_num, cfgs.device, logger)

    obs_dim, act_dim = history_len * nz, nz
    designer = SAC_Model(
        lambda: SquashedGaussianMLPActor(obs_dim, act_dim, [256, 256, 256]),
        lambda: MLPQFunction(obs_dim, act_dim, [256, 256, 256]),
        gamma=cfgs.gamma, tar_entropy=cfgs.tar_entropy, tau=cfgs.tau, device=cfgs.device
    )
    # update_itv = cfgs.n_envs if cfgs.update_itv < 0 else cfgs.update_itv
    d_trainer = AsyncGenerativeSAC_Trainer(
        cfgs.update_itv, cfgs.update_ratio, cfgs.batch_size, ReplayMem(cfgs.mem_size)
    )
    d_trainer.train(designer, env, cfgs.total_steps, res_path, cfgs.check_points)


