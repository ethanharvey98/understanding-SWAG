import argparse
import os
import random
import numpy as np
import pandas as pd
import wandb
# PyTorch
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
# Importing our custom module(s)
from posteriors.swag import SWAG
from transforms import get_mixup_cutmix
import utils_v3 as utils

# python CIFAR-10_main-Copy1.py --checkpoint_path='/cluster/home/eharve06/understanding-SWAG/experiments/swag_ImageNet_v2_torchvision/swag_epochs=30_K=20_lr=0.01_no_cov_factor=False_random_state=1001.pt' --clf_weight_decay=0.01 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/function-test' --lr_0=0.01 --model_name='test' --n=1000 --random_state=1001 --sigma=1 --tune --wandb --wandb_project='function-test'
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py') 
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('--sigma', default=1e-1, help='Variance (default: 1e-1)', type=float)
    parser.add_argument('--clf_weight_decay', default=1e-2, help='Classifier weight decay (default: 1e-2)', type=float)
    parser.add_argument('--checkpoint_path', default='', help='Path to SWAG model checkpoint (default: \'\')', type=str)
    parser.add_argument('--dataset_path', default='', help='Path to dataset (default: \'\')', type=str)
    parser.add_argument('--experiments_path', default='', help='Path to save experiments (default: \'\')', type=str)
    parser.add_argument('--k', default=5, help='Rank of low-rank covariance matrix (default: 5)', type=float)
    parser.add_argument('--lr_0', default=0.5, help='Initial learning rate (default: 0.5)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--n', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--source_num_workers', default=1, help='Number of workers (default: 1)', type=int)
    parser.add_argument('--target_num_workers', default=1, help='Number of workers (default: 1)', type=int)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether or not to log to wandb')
    parser.add_argument('--wandb_project', default='test', help='Wandb project name (default: \'test\')', type=str)    
    args = parser.parse_args()
    
    if args.wandb:
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        wandb.init(
            project = args.wandb_project,
            name = args.model_name,
            config={
                'batch_size': args.batch_size,
                'sigma': args.sigma,
                'clf_weight_decay': args.clf_weight_decay,
                'checkpoint_path': args.checkpoint_path,
                'dataset_path': args.dataset_path,
                'experiments_path': args.experiments_path,
                'k': args.k,
                'lr_0': args.lr_0,
                'model_name': args.model_name,
                'n': args.n,
                'source_num_workers': args.source_num_workers,
                'target_num_workers': args.target_num_workers,
                'save': args.save,
                'tune': args.tune,
                'random_state': args.random_state,
                'wandb': args.wandb,
                'wandb_project': args.wandb_project,
            }
        )
    
    torch.manual_seed(args.random_state)
    if args.experiments_path: utils.makedir_if_not_exist(args.experiments_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    backbone = torchvision.models.resnet50()
    
    #k = 5
    #cov_diag = torch.load('/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior/resnet50_ssl_prior_variance.pt', map_location=torch.device('cpu'))
    #cov_factor = torch.load('/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior/resnet50_ssl_prior_covmat.pt', map_location=torch.device('cpu'))
    #loc = torch.load('/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior/resnet50_ssl_prior_mean.pt', map_location=torch.device('cpu'))
    # $\Sigma = \frac{1}{2} ( \Sigma_{\text{diag}} + \Sigma_{\text{LR}} )$
    #cov_diag = (1/2)*cov_diag
    #cov_factor = np.sqrt(1/2)*cov_factor
    # $\Sigma_{\text{LR}} = \frac{1}{k-1} Q Q^T$
    #cov_factor = np.sqrt(1/(k-1))*cov_factor
    #swag_model = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(loc.to(device), cov_factor.to(device).t(), cov_diag.to(device))
    
    swag_model = SWAG(backbone)
    swag_model.load(args.checkpoint_path)
    torch.nn.utils.vector_to_parameters(swag_model.loc, backbone.parameters())
    
    backbone.fc = torch.nn.Identity()
    torch.nn.utils.vector_to_parameters(swag_model.loc, backbone.parameters())
    source_clf = torch.nn.Linear(in_features=2048, out_features=1_000, bias=True)
    target_clf = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
    model = nn.Sequential(backbone, source_clf, target_clf)
    model.to(device)
    swag_model.loc, swag_model.sq_loc, swag_model.cov_factor_columns = swag_model.loc.to(device), swag_model.sq_loc.to(device), [column.to(device) for column in swag_model.cov_factor_columns]
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': target_clf.parameters(), 'weight_decay': args.clf_weight_decay},
        {'params': backbone.parameters(), 'weight_decay': 0.0},
    ], lr=args.lr_0, momentum=0.9, nesterov=True)
    
    augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=args.dataset_path, n=args.n, tune=args.tune, random_state=args.random_state)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, num_workers=args.target_num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.target_num_workers)
    val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size, num_workers=args.target_num_workers)
    
    steps = 6000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)

    columns = ['epoch', 'train_acc1', 'train_acc5', 'train_loss', 'train_nll', 'train_prior', 'val_acc1', 'val_acc5', 'val_loss']
    model_history_df = pd.DataFrame(columns=columns)
        
    for epoch in range(epochs):
        train_metrics = utils.train_one_epoch(model, swag_model, criterion, optimizer, augmented_train_loader, lr_scheduler=lr_scheduler, sigma=args.sigma)
        val_metrics = utils.evaluate(model, criterion, val_or_test_loader)
        
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_acc1': train_metrics['acc1'],
                'train_acc5': train_metrics['acc5'],
                'train_loss': train_metrics['loss'],
                'train_nll': train_metrics['nll'],
                'train_prior': train_metrics['prior'],
                'val_acc1': val_metrics['acc1'],
                'val_acc5': val_metrics['acc5'],
                'val_loss': val_metrics['loss'],
            })
    
        row = [epoch, train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], train_metrics['nll'], train_metrics['prior'], val_metrics['acc1'], val_metrics['acc5'], val_metrics['loss']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
    
        model_history_df.to_csv(f'{args.experiments_path}/{args.model_name}.csv')
