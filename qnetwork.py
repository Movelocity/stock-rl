import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QNetwork(nn.Module):
    def __init__(self, base_dim=512, seed=0, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # base_dim 表示跟踪多少只股票

        self.qnet = nn.Sequential(
            nn.Linear(base_dim*4, base_dim*2),
            nn.BatchNorm1d(base_dim*2),
            nn.ReLU(),
            nn.Linear(base_dim*2, base_dim*1),
            nn.BatchNorm1d(base_dim*1),
            nn.ReLU(),
            nn.Linear(base_dim*1, base_dim//2),
            nn.BatchNorm1d(base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, base_dim//2),
            nn.BatchNorm1d(base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, base_dim//2),
            nn.BatchNorm1d(base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, base_dim//2),
            nn.BatchNorm1d(base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, base_dim),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )
        

    def forward(self, x):
        qvalues = self.qnet(x)
        return qvalues