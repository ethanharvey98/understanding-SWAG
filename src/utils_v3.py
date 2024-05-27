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
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.y[index]) if self.transform == None else (self.transform(self.X[index]), self.y[index])
        
def get_cifar10_datasets(root, n, tune=True, random_state=42):
    assert n in [10, 100, 1000, 10000, 50000], f'Invalid number of samples n={n}.'

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    class_indices = {cifar10_label: [idx for idx, (image, label) in enumerate(cifar10_train_dataset) if label == cifar10_label] for cifar10_label in range(10)}
    shuffled_sampled_class_indices = {cifar10_label: random_state.choice(class_indices[cifar10_label], int(n/10), replace=False) for cifar10_label in class_indices.keys()}
    
    if tune:
        if n == 10:
            mask = random_state.choice([True, True, True, True, False]*2, 10, replace=False)
            train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[mask]
            val_or_test_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[~mask]
        else:
            train_indices = {cifar10_label: shuffled_sampled_class_indices[cifar10_label][:int(4/5*int(n/10))] for cifar10_label in shuffled_sampled_class_indices.keys()}
            val_or_test_indices = {cifar10_label: shuffled_sampled_class_indices[cifar10_label][int(4/5*int(n/10)):] for cifar10_label in shuffled_sampled_class_indices.keys()}
            train_indices = np.array(list(train_indices.values())).flatten()
            val_or_test_indices = np.array(list(val_or_test_indices.values())).flatten()  
    else:
        train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()

    sampled_train_images = torch.stack([cifar10_train_dataset[index][0] for index in train_indices])
    sampled_train_labels = torch.tensor([cifar10_train_dataset[index][1] for index in train_indices])
    mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    std = torch.std(sampled_train_images, axis=(0, 2, 3))
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    
    if tune:
        val_or_test_dataset = [cifar10_train_dataset[index] for index in val_or_test_indices]
    else:
        val_or_test_dataset = cifar10_test_dataset
    
    sampled_val_or_test_images = torch.stack([image for image, label in val_or_test_dataset])
    sampled_val_or_test_labels = torch.tensor([label for image, label in val_or_test_dataset])

    augmented_train_dataset = Dataset(sampled_train_images, sampled_train_labels, augmented_transform)
    train_dataset = Dataset(sampled_train_images, sampled_train_labels, transform)
    val_or_test_dataset = Dataset(sampled_val_or_test_images, sampled_val_or_test_labels, transform)
                
    return augmented_train_dataset, train_dataset, val_or_test_dataset

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

def train_one_epoch(model, swag_model, criterion, optimizer, dataloader, lr_scheduler=None, sigma=1e-1):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0, 'nll': 0.0, 'prior': 0.0}

    for images, labels in dataloader:
        
        batch_size = len(images)
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)
        
        num_samples = 2
        state_dict = copy.deepcopy(model.state_dict())
        outputs = []
        with torch.no_grad():
            #for j in range(num_samples):
                #sampled_params = swag_model.sample()
                #torch.nn.utils.vector_to_parameters(sampled_params, model[0].parameters())
                #torch.nn.utils.vector_to_parameters(sampled_params, model[0:1].parameters())
            
            for j in torch.randint(low=0, high=10, size=(num_samples,)):
                random_states = [1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001, 9001, 10001]
                checkpoint = torch.load(f'/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2/ImageNet_v2_random_state={random_states[j]}.pt', map_location=device)['model']
                checkpoint_without_fc = {name: value for name, value in checkpoint.items() if 'fc' not in name}
                fc_checkpoint = {name[3:]: value for name, value in checkpoint.items() if 'fc' in name}
                model[0].load_state_dict(checkpoint_without_fc)
                model[1].load_state_dict(fc_checkpoint)

                embed_j = model[0](images)
                logits_j = model[1](embed_j)
                #outputs_j = torch.nn.functional.softmax(logits_j, dim=-1)
                outputs_j = logits_j
                outputs.append(outputs_j)
        outputs = torch.stack(outputs, dim=0)
        model.load_state_dict(state_dict)

        model.zero_grad()
        embed_i = model[0](images)
        logits_i = model[1](embed_i)
        #outputs_i = torch.nn.functional.softmax(logits_i, dim=-1)
        outputs_i = logits_i
        logits = model[2](embed_i)
        nll = criterion(logits, labels)
        # Note: We use the entire dataset size here.
        prior = (1/len(dataloader.dataset))*torch.logsumexp(torch.sum((-1/(2*sigma**2))*torch.sum((outputs_i-outputs)**2, dim=-1, keepdim=False), dim=-1, keepdim=False), dim=0, keepdim=False)
        loss = nll - prior
        loss.backward()
        optimizer.step()

        probas = torch.softmax(logits, dim=1)
        acc1, acc5 = accuracy(probas, labels, topk=(1, 5))
        metrics['acc1'] += batch_size/dataset_size*acc1.item()
        metrics['acc5'] += batch_size/dataset_size*acc5.item()
        metrics['loss'] += batch_size/dataset_size*loss.item()
        metrics['nll'] += batch_size/dataset_size*nll.item()
        metrics['prior'] += batch_size/dataset_size*prior.item()
        
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
                
            logits = model[2](model[0](images))
            loss = criterion(logits, labels)
            
            batch_size = len(images)
            probas = torch.softmax(logits, dim=1)
            acc1, acc5 = accuracy(probas, labels, topk=(1, 5))
            metrics['acc1'] += batch_size/dataset_size*acc1.item()
            metrics['acc5'] += batch_size/dataset_size*acc5.item()
            metrics['loss'] += batch_size/dataset_size*loss.item()
    
    return metrics