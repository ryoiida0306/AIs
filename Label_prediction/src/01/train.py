import logging
import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import DataSet, get_data_loader
from torch.utils.tensorboard import SummaryWriter  # Add this line
from torch.optim import Adam
from torch.utils import tensorboard

logger = logging.getLogger(__name__)


def train(model, train_loader, valid_loader, epochs, optimizer, save_path) :

    logger.info('training started')
    writer = SummaryWriter(log_dir = save_path)

    logging_interval = 10

    for epoch in range(epochs) :
        if epoch % logging_interval == 0 :
            logger.info('epoch: %d', epoch)
        if epoch % 100 == 0 :
            torch.save(model.state_dict(), save_path + '/model_' + str(epoch) + '.pth')
        train_loss_list = []     
        model.train()
        for batch in train_loader :
            inputs = batch[0]
            labels = batch[1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
        if epoch % logging_interval == 0 :
            logger.info('train loss: %f', np.mean(train_loss_list))
        writer.add_scalar('train_loss', np.mean(train_loss_list), epoch)

        if epoch % 100 == 0 :
            model.eval()
            valid_loss_list = []
            valid_accuracy_list = []
            for batch in valid_loader :
                inputs = batch[0]
                labels = batch[1]
                outputs = model(inputs)
                loss = model.loss(outputs, labels)
                valid_loss_list.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / len(labels)
                valid_accuracy_list.append(accuracy)
            logger.info('valid loss: %f', np.mean(valid_loss_list))
            logger.info('valid accuracy: %f', np.mean(valid_accuracy_list))
            writer.add_scalar('valid_loss', np.mean(valid_loss_list), epoch)
            writer.add_scalar('valid_accuracy', np.mean(valid_accuracy_list), epoch)
        
    logger.info('training finished')
        
