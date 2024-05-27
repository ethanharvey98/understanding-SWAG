import argparse
import time
# PyTorch
import torch
import torchvision

# python random_kitchen_sinks.py --D=1000 --gamma=1e2 --weight_decay=1e2
def fit_kitchen_sink(model, dataloader, D, gamma, weight_decay):
    # Approximates Gaussian Process regression
    #     with Gaussian kernel of variance gamma^2
    # weight_decay: regularization parameter
    # dataset: X is dxN, y is 1xN
    # test: xtest is dx1
    # D: dimensionality of random feature
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
    
    w, b = torch.randn(D, 3*224*224).to(device), 2 * torch.pi * torch.rand(D, 1).to(device)
    y, m, Phi = [], [], []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(dataloader):
            print(f'Batch {batch_index} of {len(dataloader)}')
            y.append(labels)

            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)

            X = images.view(images.shape[0], -1).t()
            d, N = X.shape
            logits = model(images)
            probas = torch.nn.functional.softmax(logits, dim=-1)
            
            if device.type == 'cuda':
                probas = probas.cpu()
            
            m.append(probas)
            
            Z = torch.cos(gamma * w @ X + b * torch.ones(1, N).to(device))
            phi = torch.stack([torch.linalg.solve(weight_decay * torch.eye(D).to(device) + Z @ Z.t(), Z * (labels==label).long().view(1, -1)) for label in range(1_000)], dim=-1)
            
            if device.type == 'cuda':
                phi = phi.cpu()
            
            Phi.append(phi)
            
        if device.type == 'cuda':
            w, b = w.cpu(), b.cpu()

    return w, b, torch.cat(y, dim=0), torch.cat(m, dim=0), torch.cat(Phi, dim=1)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--D', default=20, help='D (default: 20)', type=int)
    parser.add_argument('--gamma', default=1e-2, help='Gamma (default: 1e-2)', type=float)
    parser.add_argument('--num_workers', default=16, help='Number of workers (default: 16)', type=int)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--weight_decay', default=1e-2, help='Lambda (default: 1e-2)', type=float)
    args = parser.parse_args()
    
    torch.manual_seed(args.random_state)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=176),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=train_transform)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=args.num_workers)
    
    # Note: Use train subset for testing.
    N = 10_000
    subset_indices = torch.arange(start=0, end=len(train_dataset))[:N]
    subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=128)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    
    start_time = time.time()
    print(f'Start time: {start_time} seconds')
    w, b, y, m, Phi = fit_kitchen_sink(model=model, dataloader=train_loader, D=args.D, gamma=args.gamma, weight_decay=args.weight_decay)
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    
    print(w.shape)
    print(b.shape)
    print(y.shape)
    print(m.shape)
    print(Phi.shape)
    
    torch.save({
        'w': w,
        'b': b,
        'y': y,
        'm': m,
        'Phi': Phi,
    }, f'./random_kitchen_sinks_gamma={args.gamma}_weight_decay={args.weight_decay}_random_state={args.random_state}.pt')