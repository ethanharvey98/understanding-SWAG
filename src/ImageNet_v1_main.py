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
import utils
from transforms import get_mixup_cutmix

class CyclicLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_0, T, M, last_epoch=-1):
        self.lr_0 = lr_0
        self.T = T
        self.M = M
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch // (self.T // self.M)
        batch_idx = self.last_epoch % (self.T // self.M)
        rcounter = epoch * (self.T // self.M) + batch_idx
        cos_inner = np.pi * (rcounter % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        return [lr for _ in self.optimizer.param_groups]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--epochs', default=90, help='Number of epochs (default: 600)', type=int)
    parser.add_argument('--experiments_path', default='', help='Path to save experiments (default: \'\')', type=str)
    parser.add_argument('--cycles', default=1, help='Number of cycles (default: 1)', type=int)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
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
                'cycles': args.cycles,
                'model_name': args.model_name,
                'num_workers': args.num_workers,
                'random_state': args.random_state,
                'wandb': args.wandb,
                'wandb_project': args.wandb_project,
            }
        )
        
    torch.manual_seed(args.random_state)
    if args.experiments_path: utils.makedir_if_not_exist(args.experiments_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights).to(device)
    #model = torchvision.models.resnet50().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
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
        
    #augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)
    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=args.num_workers)
    
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    T, M = args.epochs, args.cycles
    #lr_scheduler = CyclicLR(optimizer, 0.01, T, M)
    lr_scheduler = CyclicLR(optimizer, 0.01, T*len(augmented_train_loader), M)
    
    
    last_epoch = -1
    if os.path.exists(f'{args.experiments_path}/{args.model_name}.pt'):
        checkpoint = torch.load(f'{args.experiments_path}/{args.model_name}.pt')
        model_history_df = pd.read_csv(f'{args.experiments_path}/{args.model_name}.csv', index_col=0)
        last_epoch = checkpoint['epoch']
        torch.set_rng_state(checkpoint['random_state'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        columns = ['epoch', 'train_acc1', 'train_acc5', 'train_loss', 'val_acc1', 'val_acc5', 'val_loss']
        model_history_df = pd.DataFrame(columns=columns)
        
    for epoch in range(last_epoch+1, args.epochs):
        
        #train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader)
        #lr_scheduler.step()
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader, lr_scheduler)
        #train_metrics = utils.train_one_epoch(model, criterion, optimizer, train_loader)
        val_metrics = utils.evaluate(model, criterion, val_loader)
        
        if epoch%(T//M) == (T//M)-1:
            torch.save({
                'epoch': epoch,
                'random_state': torch.get_rng_state(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, f'{args.experiments_path}/{args.model_name}_{epoch//(T//M)}.pt')
            

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'lr': lr_scheduler.get_last_lr()[0],
                'train_acc1': train_metrics['acc1'],
                'train_acc5': train_metrics['acc5'],
                'train_loss': train_metrics['loss'],
                'val_acc1': val_metrics['acc1'],
                'val_acc5': val_metrics['acc5'],
                'val_loss': val_metrics['loss'],
            })
            
        row = [epoch, train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], val_metrics['acc1'], val_metrics['acc5'], val_metrics['loss']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_path}/{args.model_name}.csv')
        
        torch.save({
            'epoch': epoch,
            'random_state': torch.get_rng_state(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, f'{args.experiments_path}/{args.model_name}.pt')