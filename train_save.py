import os

import torch
import numpy as np
import pandas as pd
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True,
    )
    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = ToTensor()
    )

    from torch.utils.data import DataLoader
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                              batch_size=100,
                                              shuffle=True),

        'test': torch.utils.data.DataLoader(test_data,
                                              batch_size=100,
                                              shuffle=True),
    }

    from torch import optim

    cnn = CNN()
    cnn = cnn.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

    from torch.autograd import Variable
    num_epochs = 10
    def train(num_epochs, cnn, loaders):

        cnn.train()

        # Train the model
        total_step = len(loaders['train'])

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                images = images.to(device)
                labels = labels.to(device)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = cnn(b_x)[0]
                loss = loss_func(output, b_y)

                # clear gradients for this training step
                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass

            pass


        pass

    train(num_epochs, cnn, loaders)

    torch.save(cnn.state_dict(), "./model.pth")

if __name__ == "__main__":
    main()