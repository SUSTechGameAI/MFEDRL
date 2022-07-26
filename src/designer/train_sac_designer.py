# """
#   @Time : 2021/11/29 20:03 
#   @Author : Ziqi Wang
#   @File : train_sac_designer.py 
# """
# 
# import os
# import importlib
# from root import PRJROOT
# from src.algo.sac import SAC_Model, SAC_Trainer
# from src.algo.ac_model import SquashedGaussianMLPActor, MLPQFunction
# from src.environment.env import make_vec_generation_env
# from src.algo.replay_memory import ReplayMem
# from src.utils import auto_dire
# from src.gan.gan_config import nz
# 
# 
# def set_parser(parser):
#     parser.add_argument(
#         '--n_envs', type=int, default=8,
#         help='Number of parallel environments.'
#     )
#     parser.add_argument(
#         '--max_seg_num', type=int, default=100,
#         help='Maximum nubmer of segments to generate in the generation enviroment.'
#     )
#     parser.add_argument(
#         '--archive_len', type=int, default=8,
#         help='The length of archive for previous segments.'
#     )
#     parser.add_argument(
#         '--total_steps', type=int, default=int(1e5),
#         help='Total time steps (frames) for training PPO designer.'
#     )
#     parser.add_argument('--gamma', type=float, default=0.75)
#     parser.add_argument('--tar_entropy', type=float, default=-nz)
#     parser.add_argument('--tau', type=float, default=0.005)
#     parser.add_argument('--update_itv', type=int, default=500)
#     parser.add_argument('--update_repeats', type=int, default=100)
#     parser.add_argument('--batch_size', type=int, default=384)
#     parser.add_argument('--mem_size', type=int, default=int(1e6))
#     parser.add_argument(
#         '--device', type=str, default='cuda:0',
#         help='Device for training the DDPG agent.'
#     )
#     parser.add_argument(
#         '--rfunc_name', type=str, default='default',
#         help='Name of the file where the reward function located. '
#              'The file must be put in the \'src.reward_functions\' package.'
#     )
#     parser.add_argument(
#         '--res_path', type=str, default='',
#         help='Path relateed to \'/exp_data\'to save the training log. '
#              'If not specified, a new folder named exp{id} will be created.'
#     )
#     parser.add_argument(
#         '--check_points', type=int, nargs='+',
#         help='check points to save deisigner, specified by the number of time steps.'
#     )
# 
# def train_designer(cfgs):
#     # torch.set_default_dtype(torch.float)
#     if not cfgs.res_path:
#         cfgs.res_path = auto_dire(PRJROOT + 'exp_data')
#     else:
#         cfgs.res_path = PRJROOT + 'exp_data/' + cfgs.res_path
#         os.makedirs(cfgs.res_path, exist_ok=True)
# 
#     rfunc = (
#         importlib.import_module('src.environment.rfuncs')
#         .__getattribute__(f'{cfgs.rfunc_name}')
#     )
#     with open(cfgs.res_path + '/run_config.txt', 'w') as f:
#         f.write('---------SAC---------\n')
#         args_strlines = [
#             f'{key}={val}\n' for key, val in vars(cfgs).items()
#             if key not in {'rfunc_name', 'res_path', 'entry', 'check_points'}
#         ]
#         f.writelines(args_strlines)
#         f.write('-' * 50 + '\n')
#         f.write(str(rfunc))
# 
#     env = make_vec_generation_env(
#         cfgs.n_envs, rfunc, cfgs.res_path, cfgs.max_seg_num, cfgs.archive_len,
#         log_itv=100, device=cfgs.device, log_targets=['file']
#     )
# 
#     obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
#     designer = SAC_Model(
#         lambda: SquashedGaussianMLPActor(obs_dim, act_dim, [128, 256]),
#         lambda: MLPQFunction(obs_dim, act_dim, [128, 256]),
#         gamma=cfgs.gamma, tar_entropy=cfgs.tar_entropy, tau=cfgs.tau, device=cfgs.device
#     )
#     d_trainer = SAC_Trainer(
#         env, cfgs.total_steps, cfgs.update_itv, cfgs.update_repeats, max(cfgs.batch_size, cfgs.max_seg_num * cfgs.n_envs),
#         cfgs.batch_size, ReplayMem(cfgs.mem_size), cfgs.res_path, cfgs.check_points
#     )
#     d_trainer.train(designer)
# 
