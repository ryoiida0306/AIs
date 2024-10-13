import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module) :

    def __init__(self) :
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)



    def forward(self, x) :
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.log_softmax(x)
        return x
    
    def loss(self, output, target):
        long_target = target.clone().detach().long()
        return nn.CrossEntropyLoss()(output, long_target)
    
