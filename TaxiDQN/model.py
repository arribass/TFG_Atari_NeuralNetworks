"""
    Modelo DQN - Paquete TaxiDQN
    Adrián Arribas
    UPNA 
"""
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    ''' Implementacion del modelo DQN '''
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x