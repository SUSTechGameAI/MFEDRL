"""
  @Time : 2022/1/4 10:15 
  @Author : Ziqi Wang
  @File : use_designer.py 
"""

import torch
from src.utils.filesys import get_path


class Designer:
    def __init__(self, model_path, device='cpu'):
        safe_model_path = get_path(model_path)
        self.model = torch.load(safe_model_path, map_location=device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = device

    def act(self, obs):
        model_in = torch.tensor(obs, device=self.device)
        if len(obs.shape) == 1:
            model_in = model_in.unsqueeze(0)
        model_output, _ = self.model(model_in)
        return model_output.squeeze().cpu().numpy()


