import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QNetwork(nn.Module):
    def __init__(self, base_dim=512, seed=0):
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

class QNetworkE(nn.Module):
    def __init__(self, base_dim=512, seed=0, emb_size=1024):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # base_dim 表示跟踪多少只股票

        self.read_price = nn.Sequential(
            nn.Linear(base_dim*4, base_dim*2),
            nn.BatchNorm1d(base_dim*2),
            nn.ReLU(),
            nn.Linear(base_dim*2, base_dim*1),
            nn.BatchNorm1d(base_dim*1),
            nn.ReLU(),
            nn.Linear(base_dim*1, base_dim//2),
            nn.BatchNorm1d(base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, base_dim),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(),
        )

        self.read_emb = nn.Sequential(
            nn.Linear(emb_size, base_dim),
            nn.ReLU()
        )
        self.post_net = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
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
        

    def forward(self, hist_price, embed):
        p = self.read_price(hist_price)
        e = self.read_emb(embed)
        qvalues = self.post_net(p+e)
        return qvalues