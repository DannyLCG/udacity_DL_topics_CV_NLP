import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms


class Net(nn.Module):
    '''Define a custom CNN w. its forward pass.'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #input, output, kernel_size, stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        '''Define forward pass'''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) #input tensor, kernel_size
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        output = F.log_softmax(x, 1)

        return output

def train(model, train_loader, optimizer, epoch):
    '''Training function'''
    # 1. Loop through data
    for batch_idx, (data, target) in enumerate(train_loader):
        # 2. Zero all the gradients before the backward pass
        optimizer.zero_grad()
        # 3. Forward pass
        output = model(data)
        # 4. Calculate loss
        loss = F.nll_loss(output, target)
        # 5. Backpropagation: compute gradient of the loss with respect to model parameters
        loss.backward()
        # 6. update the weights
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(), #REMOVE COMA?
                )
            )


def test(model, test_loader):
    '''Testing function'''
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Disable gradient calculation
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="Number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="Learning rate (default: 1.0)"
    )

    args = parser.parse_args()

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}

    # Instance a torch transformation to cast input data to a tensor
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    
    # Add the CIFAR10 dataset and create your data loaders
    train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10("../data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Instance the CNN model
    model = Net()

    # Instance an optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Train and test the model for the given number of epochs
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)

    #torch.save(model.state_dict(), "models/mnist_cnn.pt")

if __name__ == "__main__":
    main()
