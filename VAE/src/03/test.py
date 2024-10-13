import logging
import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import DataSet, get_data_loader
from torch.utils.tensorboard import SummaryWriter  # Add this line
from torch.optim import Adam

logger = logging.getLogger(__name__)

def test(model, test_loader) :

    logger.info('testing started')
    model.eval()
    test_accuracy_list = []
    for batch in test_loader :
        inputs = batch[0]
        labels = batch[1]
        target = inputs
        outputs, mu, log_var = model(inputs)
        loss = model.loss(outputs, target, mu, log_var)
        test_accuracy_list.append(loss.item())
        
        # outputs = model(inputs)
        # _, predicted = torch.max(outputs, 1)
        # correct = (predicted == labels).sum().item()
        # accuracy = correct / len(labels)
        # test_accuracy_list.append(accuracy)
    logger.info('test accuracy: %f', np.mean(test_accuracy_list))
    logger.info('testing finished')