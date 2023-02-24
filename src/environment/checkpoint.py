"""
  @Time : 2021/11/11 22:56 
  @Author : Ziqi Wang
  @File : checkpoint.py 
"""

import os
from stable_baselines3.common.callbacks import BaseCallback


class MyCheckpointCallback(BaseCallback):
    """
    Callback for saving ztraces model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_path, checkpoints):
        super(MyCheckpointCallback, self).__init__(0)
        self.save_path = save_path
        self.check_points = set(checkpoints)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps in self.check_points:
            path = os.path.join(self.save_path, f"designer_at_{self.num_timesteps}")
            self.model.save(path)
            print(f"Designer checkpoint at {self.num_timesteps} step saved")
        return True
