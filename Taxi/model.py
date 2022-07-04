import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()

    def forward(self, x):
        return x