import os
import copy
import numpy as np
# PyTorch
import torch
import torchvision
import torchmetrics

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        
def flatten_params(model, deepcopy=True):
    if deepcopy: model = copy.deepcopy(model)
    return torch.cat([param.detach().view(-1) for param in model.parameters()])

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
        
def bn_update(model, dataloader):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda(non_blocking=True)
            images_var = torch.autograd.Variable(images)
            b = images_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(images_var)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}

    for images, labels in dataloader:
                        
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        probabilities = torch.softmax(logits, dim=1)
        acc1, acc5 = accuracy(probabilities, labels, topk=(1, 5))
        metrics['acc1'] += batch_size/dataset_size*acc1.item()
        metrics['acc5'] += batch_size/dataset_size*acc5.item()
        metrics['loss'] += batch_size/dataset_size*loss.item()
        
        if lr_scheduler:
            lr_scheduler.step()
            
    return metrics

def evaluate(model, criterion, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0}
            
    with torch.no_grad():
        for images, labels in dataloader:
                        
            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)
                
            logits = model(images)
            loss = criterion(logits, labels)
            
            batch_size = len(images)
            probabilities = torch.softmax(logits, dim=1)
            acc1, acc5 = accuracy(probabilities, labels, topk=(1, 5))
            metrics['acc1'] += batch_size/dataset_size*acc1.item()
            metrics['acc5'] += batch_size/dataset_size*acc5.item()
            metrics['loss'] += batch_size/dataset_size*loss.item()
    
    return metrics