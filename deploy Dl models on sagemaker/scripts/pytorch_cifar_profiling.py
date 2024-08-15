import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from smdebug import modes
from smdebug.profiler.utils import str2bool
import smdebug.pytorch as smd

def train(args, net, device):
    # ====================================#
    # 1. Create the hook
    # ====================================#
    hook = smd.get_hook(create_if_not_exists=True)
    batch_size = args.batch_size
    epoch = args.epoch

    # Instance data transformations
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Download datasets
    train_dataset = torchvision.datasets.CIFAR10( root="./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    
    # Instance data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    epoch_times = []
    # ======================================================#
    # 2. Set hook to track the loss 
    # ======================================================#
    if hook:
        hook.register_loss(loss_optim)

    # Define training loop
    for i in range(epoch):
        print("START TRAINING")
        # ======================================================#
        # 3. Set hook to training mode
        # ======================================================#
        if hook:
            hook.set_mode(modes.TRAIN)
        start = time.time()
        # Set model in training mode
        net.train()
        train_loss = 0
        # 1. Loop through data
        for _, (inputs, targets) in enumerate(train_loader):
            # Send input tensors to the specified device
            inputs, targets = inputs.to(device), targets.to(device)
            # 2. Zero all the gradients befor backward pass
            optimizer.zero_grad()
            # 3. Forward pass
            outputs = net(inputs)
            # 4. Compute the loss
            loss = loss_optim(outputs, targets)
            # 5. Backprop: compute radients with respect to model parameters
            loss.backward()
            # 6. Backprop: update weights
            optimizer.step()
            train_loss += loss.item()

        print("START VALIDATING")
        # ======================================================#
        # 4. Set hook to eval mode
        # ======================================================#
        if hook:
            hook.set_mode(modes.EVAL)
        # Set model to evaluation mode
        net.eval()
        val_loss = 0
        with torch.no_grad(): # Disable gradient calculation
            # 1.Lop through data
            for _, (inputs, targets) in enumerate(test_loader):
                # Set inputs to the specified device
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = loss_optim(outputs, targets)
                val_loss += loss.item()
        
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print(
            "Epoch %d: train loss %.3f, val loss %.3f, in %.1f sec"
            % (i, train_loss, val_loss, epoch_time)
        )

    # Calculate training time
    p50 = np.percentile(epoch_times, 50)
    return p50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
        )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1
        )
    parser.add_argument(
        "--gpu",
        type=str2bool,
        default=True
        )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50"
        )

    opt = parser.parse_args()

    for key, value in vars(opt).items():
        print(f"{key}:{value}")
    # Create model
    net = models.__dict__[opt.model](pretrained=True)
    if opt.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)

    # Start the training
    median_time = train(opt, net, device)
    print("Median training time per Epoch=%.1f sec" % median_time)


if __name__ == "__main__":
    main()
