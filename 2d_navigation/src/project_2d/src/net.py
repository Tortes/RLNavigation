import torch
import numpy as np
from torch import nn

from tianshou.data import to_torch

class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 device='cpu', hidden_layer_size=128):
        super().__init__()
        self.device = device
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.conv1 = torch.nn.Conv1d(1, 32, 5, stride=2)
        self.conv2 = torch.nn.Conv1d(32, 32, 3, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32*126, 256)
        self.fc2 = nn.Linear(260, 128)
        self.relu = nn.ReLU()

    def forward(self, s, state=None, info={}):
        bs = len(s.scan_range)
        sd = s.scan_range.reshape((bs, 1, -1))
        sd = to_torch(sd, device=self.device, dtype=torch.float32)
        sd = self.relu(self.bn1(self.conv1(sd)))
        sd = self.relu(self.bn2(self.conv2(sd)))
        sd = torch.flatten(sd,1)
        sd = self.fc1(sd)
        
        si = to_torch(s.state_info, device=self.device, dtype=torch.float32)
        sd = torch.cat((sd, si), 1)
        sd = self.fc2(sd)

        return sd, state

