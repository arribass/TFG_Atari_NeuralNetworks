import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        return x