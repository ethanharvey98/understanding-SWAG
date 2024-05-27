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
from transforms import get_mixup_cutmix
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--epochs', default=600, help='Number of epochs (default: 600)', type=int)
    parser.add_argument('--experiments_path', default='', help='Path to save experiments (default: \'\')', type=str)
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
                'model_name': args.model_name,
                'num_workers': args.num_workers,
                'random_state': args.random_state,
                'wandb': args.wandb,
            }
        )
        
    torch.manual_seed(args.random_state)
    if args.experiments_path: utils.makedir_if_not_exist(args.experiments_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet50().to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    norm_parameters = [param for name, param in model.named_parameters() if 'bn' in name]
    other_parameters = [param for name, param in model.named_parameters() if 'bn' not in name]
    optimizer = torch.optim.SGD([
        {'params': norm_parameters, 'weight_decay': 0.0},
        {'params': other_parameters, 'weight_decay': 2e-05},
    ], lr=0.5, momentum=0.9)
    
    augmented_train_transform = torchvision.transforms.Compose([
        # 1. Random resize and crop trianing images to 176.
        torchvision.transforms.RandomResizedCrop(size=(176, 176)),
        # 2. For all setups we normalize the images by training set 
        #    mean and standard deviation after the application of all augmentation.
        torchvision.transforms.TrivialAugmentWide(),
        torchvision.transforms.ToTensor(),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.RandomErasing(p=0.1),
    ])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize training images to 176.
        torchvision.transforms.Resize(size=(176, 176)),
        # 2. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize validation images to 232.
        torchvision.transforms.Resize(size=(232, 232)),
        # 2. Crop validation images to 224.
        torchvision.transforms.CenterCrop(size=(224, 224)),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augmented_train_dataset = datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=augmented_train_transform)
    train_dataset = datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/val/', transform=val_transform)
    
    num_classes = len(augmented_train_dataset.classes)
    mixup_cutmix = get_mixup_cutmix(mixup_alpha=0.2, cutmix_alpha=1.0, num_categories=num_classes)
    
    def collate_fn(batch):
        return mixup_cutmix(*torch.utils.data.dataloader.default_collate(batch))
    
    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=args.num_workers)
    
    linear_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: np.linspace(0.01, 1.0, num=6)[epoch])
    cyclical_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-5)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_lr_scheduler, cyclical_lr_scheduler], [5])
    
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
        
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader)
        lr_scheduler.step()
        #train_metrics = utils.train_one_epoch(model, criterion, optimizer, train_loader)
        val_metrics = utils.evaluate(model, criterion, val_loader)

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'norm_lr': lr_scheduler.get_last_lr()[0],
                'other_lr': lr_scheduler.get_last_lr()[1],
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