import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module) :

    def __init__(self) :
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3 , padding=0) # 28 x 28 x 1 -> 26 x 26 x 32
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 26 x 26 x 32 -> 13 x 13 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=0) # 13 x 13 x 32 -> 10 x 10 x 64
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 10 x 10 x 64 -> 5 x 5 x 64
        self.flatten = nn.Flatten() # 5 x 5 x 64 -> 1600
        self.linear1 = nn.Linear(64*5*5, 128) # 1600 -> 128
        self.relu = nn.ReLU() # 128 -> 128
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
    def loss(self, output, target):
        long_target = target.clone().detach().long()
        return nn.CrossEntropyLoss()(output, long_target)
    
