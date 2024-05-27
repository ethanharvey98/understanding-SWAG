# PyTorch
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=20),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=20, out_features=2),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LRMVNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=1),
        )
        self.cov_factor = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=1),
        )
        self.cov_diag = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=1),
        )
        
    def forward(self, x):
        loc, cov_factor, cov_diag = self.loc(x), self.cov_factor(x), self.cov_diag(x)
        return loc, cov_factor, cov_diag
    
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x