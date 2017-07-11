
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Estimator(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.conv_1 = nn.Conv2d(4, 16, (8,8), 4)
        self.conv_2 = nn.Conv2d(16, 32, (4,4), 2)
        self.dense_1 = nn.Linear(32 * 9 * 9, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self,x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.dense_1(x))
        x = self.out(x)
        return x

class LinearEstimator(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.dense_1 = nn.Linear(4, 64)
        self.out = nn.Linear(64, num_actions)

    def forward(self,x):
        x = F.relu(self.dense_1(x))
        x = self.out(x)
        return x
