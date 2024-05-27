import copy
import numpy as np
# PyTorch
import torch
import torchvision
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

def flatten_params(model, deepcopy=True):
    if deepcopy: model = copy.deepcopy(model)
    return torch.cat([param.detach().view(-1) for param in model.parameters()])

class SWAG():
    def __init__(self, base_model, no_cov_factor=False, max_num_models=20):
        self.loc = flatten_params(base_model)
        self.sq_loc = flatten_params(base_model)**2
        self.no_cov_factor = no_cov_factor
        self.max_num_models = max_num_models
        self.cov_factor_columns = []
        self.n_models = 0
        
    @property
    def cov_diag(self):
        return torch.max(torch.tensor(1e-6), self.sq_loc-self.loc**2)
    
    @property
    def cov_factor(self):
        if self.no_cov_factor:
            raise ValueError('Covariance factor cannot be computed when \'no_cov_factor\' is True.')
        else:
            return torch.max(torch.tensor(1e-6), torch.stack(self.cov_factor_columns).t())
    
    @property
    def K(self):
        return min(self.n_models, self.max_num_models)
    
    @property
    def mvn(self):
        if self.no_cov_factor:
            # TODO: Is there a better way to create a diagonal multivariate normal?
            return LowRankMultivariateNormal(self.loc, torch.zeros(len(self.loc), 1), self.cov_diag)
        else:
            return LowRankMultivariateNormal(self.loc, (1/np.sqrt(2*(self.K-1)))*self.cov_factor, (1/2)*self.cov_diag)
    
    def collect_model(self, model):
        flattened_params = flatten_params(model)
        self.loc = self.loc * self.n_models/(self.n_models + 1) + flattened_params/(self.n_models + 1)
        self.sq_loc = self.sq_loc * self.n_models/(self.n_models + 1) + (flattened_params**2)/(self.n_models + 1)
        if self.no_cov_factor is False:
            if len(self.cov_factor_columns) >= self.max_num_models:
                self.cov_factor_columns.pop(0)
            self.cov_factor_columns.append(flattened_params-self.loc)
        self.n_models += 1
        
    def sample(self):
        return self.mvn.sample()
    
    def log_prob(self, tensor_or_module):
        if isinstance(tensor_or_module, torch.Tensor):
            return self.mvn.log_prob(tensor_or_module)
        elif isinstance(tensor_or_module, torch.nn.Module):
            return self.mvn.log_prob(flatten_params(tensor_or_module))
    
    def save(self, path):
        torch.save({
            'loc': self.loc,
            'sq_loc': self.sq_loc,
            'no_cov_factor': self.no_cov_factor,
            'max_num_models': self.max_num_models,
            'cov_factor_columns': self.cov_factor_columns,
            'n_models': self.n_models,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.loc = checkpoint['loc']
        self.sq_loc = checkpoint['sq_loc']
        self.no_cov_factor = checkpoint['no_cov_factor']
        self.max_num_models = checkpoint['max_num_models']
        self.cov_factor_columns = checkpoint['cov_factor_columns']
        self.n_models = checkpoint['n_models']