import argparse
import os
import random
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
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--checkpoint_path', default='', help='Path to saved checkpoint (default: \'\')', type=str)
    parser.add_argument('--epochs', default=30, help='Number of epochs (default: 30)', type=int)
    parser.add_argument('--experiments_path', default='', help='Path to save experiments (default: \'\')', type=str)
    parser.add_argument('--K', default=20, help='K (default: 20)', type=int)
    parser.add_argument('--lr', default=10e-3, help='Constant learning rate (default: 10e-3)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--no_cov_factor', action='store_true', default=False, help='Whether or not to use a low-rank covariance (default: False)')
    parser.add_argument('--num_workers', default=16, help='Number of workers (default: 16)', type=int)
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
                'epochs': args.epochs,
                'experiments_path': args.experiments_path,
                'K': args.K,
                'lr': args.lr,
                'model_name': args.model_name,
                'no_cov_factor': args.no_cov_factor,
                'num_workers': args.num_workers,
                'random_state': args.random_state,
                'wandb': args.wandb,
            }
        )
        
    #torch.manual_seed(args.random_state)
    if args.experiments_path: utils.makedir_if_not_exist(args.experiments_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    #model = torchvision.models.resnet50(weights=weights).to(device)
    model = torchvision.models.resnet50()
    checkpoint = torch.load(args.checkpoint_path)
    torch.set_rng_state(checkpoint['random_state'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    augmented_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Random resize and crop trianing images to 224.
        torchvision.transforms.RandomResizedCrop(size=(224, 224)),
        # 2. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize training images to 224.
        torchvision.transforms.Resize(size=(224, 224)),
        # 2. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize validation images to 224.
        torchvision.transforms.Resize(size=(224, 224)),
        # 2. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augmented_train_dataset = datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=augmented_train_transform)
    train_dataset = datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/val/', transform=val_transform)
        
    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=args.num_workers)
        
    columns = ['epoch', 'train_acc1', 'train_acc5', 'train_loss', 'val_acc1', 'val_acc5', 'val_loss']
    model_history_df = pd.DataFrame(columns=columns)
    
    swag = SWAG(model, no_cov_factor=args.no_cov_factor, max_num_models=args.K)
    swag.collect_model(model)
    
    train_metrics = utils.evaluate(model, criterion, augmented_train_loader)
    #train_metrics = utils.evaluate(model, criterion, train_loader)
    val_metrics = utils.evaluate(model, criterion, val_loader)

    if args.wandb:
        wandb.log({
            'epoch': 0,
            'train_acc1': train_metrics['acc1'],
            'train_acc5': train_metrics['acc5'],
            'train_loss': train_metrics['loss'],
            'val_acc1': val_metrics['acc1'],
            'val_acc5': val_metrics['acc5'],
            'val_loss': val_metrics['loss'],
        })

    row = [0, train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], val_metrics['acc1'], val_metrics['acc5'], val_metrics['loss']]
    model_history_df.loc[0] = row
    print(model_history_df.iloc[0])
        
    for epoch in range(1, args.epochs+1):
        
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader)
        #train_metrics = utils.train_one_epoch(model, criterion, optimizer, train_loader)
        flattened_params = utils.flatten_params(model)
        swag.collect_model(model)
        sampled_params = swag.loc
        torch.nn.utils.vector_to_parameters(sampled_params, model.parameters())
        utils.bn_update(model, train_loader)
        swag_metrics = utils.evaluate(model, criterion, val_loader)
        torch.nn.utils.vector_to_parameters(flattened_params, model.parameters())

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_acc1': train_metrics['acc1'],
                'train_acc5': train_metrics['acc5'],
                'train_loss': train_metrics['loss'],
                'val_acc1': swag_metrics['acc1'],
                'val_acc5': swag_metrics['acc5'],
                'val_loss': swag_metrics['loss'],
            })
            
        row = [epoch, train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], swag_metrics['acc1'], swag_metrics['acc5'], swag_metrics['loss']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_path}/{args.model_name}.csv')
        
    swag.save(f'{args.experiments_path}/{args.model_name}.pt')