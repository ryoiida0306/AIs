import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module) :

    def __init__(self) :
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5 , padding=0) # 28 x 28 x 1 -> 24 x 24 x 3
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 24 x 24 x 3 -> 12 x 12 x 3
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5, padding=0) # 12 x 12 x 3 -> 8 x 8 x 6
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 8 x 8 x 6 -> 4 x 4 x 6
        self.flatten = nn.Flatten() # 4 x 4 x 6 -> 96
        self.relu = nn.ReLU()
        self.linear = nn.Linear(96, 32) # 96 -> 32
        self.linear2 = nn.Linear(32, 10)
        
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x) :
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x
    
    def loss(self, output, target):
        long_target = target.clone().detach().long()
        return nn.CrossEntropyLoss()(output, long_target)
    
    
