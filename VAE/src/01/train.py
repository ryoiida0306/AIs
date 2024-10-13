import logging
import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import DataSet, get_data_loader
from plot import save_tensors_as_pdf
from torch.utils.tensorboard import SummaryWriter  # Add this line
from torch.optim import Adam
from torch.utils import tensorboard
import matplotlib.pyplot as plt

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
        train_loss_kl_list = []
        train_loss_reconstruction_list = []
        inputs_example = torch.FloatTensor()
        outputs_example = torch.FloatTensor()
        model.train()
        for batch in train_loader :
            inputs = batch[0]
            labels = batch[1]
            target = inputs
            optimizer.zero_grad()
            outputs, mu, log_var = model(inputs)
            loss = model.loss(outputs, target, mu, log_var)
            loss_kl = model.loss_kl_divergence(mu, log_var)
            loss_reconstruction = model.loss_reconstruction(outputs, target)
            # logger.info('loss: %f', loss.item())
            # logger.info('kl: %f', loss_kl.item())
            # logger.info('reconstruction: %f', loss_reconstruction.item())
            # plt.imshow(inputs[0].reshape(28, 28), cmap='gray')
            # plt.show()
            # plt.imshow(outputs[0].detach().cpu().numpy().reshape(28, 28), cmap='gray')
            # plt.show()
            inputs_example = inputs
            outputs_example = outputs
            if loss.item() != loss.item() :
                logger.info('nan detected')
                logger.info('train_loss_list: %s', train_loss_list)
                logger.info('train_loss_kl_list: %s', train_loss_kl_list)
                logger.info('mu: %s', mu)
                logger.info('log_var: %s', log_var)
                logger.info('outputs_max: %s', torch.max(outputs))
                logger.info('outputs_min: %s', torch.min(outputs))
                plt.imshow(inputs[0].reshape(28, 28), cmap='gray')
                plt.show()
                plt.imshow(outputs[0].detach().cpu().numpy().reshape(28, 28), cmap='gray')
                plt.show()
                return 0
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            train_loss_kl_list.append(loss_kl.item())
            train_loss_reconstruction_list.append(loss_reconstruction.item())
        if epoch % logging_interval == 0 :
            logger.info('train loss: %f', np.mean(train_loss_list))
            logger.info('train loss kl: %f', np.mean(train_loss_kl_list))
            logger.info('train loss reconstruction: %f', np.mean(train_loss_reconstruction_list))
            # logger.info('train loss kl: %s', str(train_loss_kl_list))
            # logger.info('train loss reconstruction: %s', str(train_loss_reconstruction_list))
        writer.add_scalar('train_loss', np.mean(train_loss_list), epoch)

        if epoch % 100 == 0 :
            model.eval()
            # plt.imshow(inputs_example[0].reshape(28, 28), cmap='gray')
            # plt.show()
            # plt.imshow(outputs_example[0].detach().cpu().numpy().reshape(28, 28), cmap='gray')
            # plt.show()
            save_tensors_as_pdf(inputs_example[0], outputs_example[0], epoch)

            valid_loss_list = []
            valid_accuracy_list = []
            for batch in valid_loader :
                inputs = batch[0]
                labels = batch[1]
                target = inputs
                outputs, mu, log_var = model(inputs)
                loss = model.loss(outputs, target, mu, log_var)
                valid_loss_list.append(loss.item())
                # _, predicted = torch.max(outputs, 1)
                # correct = (predicted == labels).sum().item()
                # accuracy = correct / len(labels)
                # valid_accuracy_list.append(accuracy)

                # outputs = model(inputs)
                # loss = model.loss(outputs, labels)
                # valid_loss_list.append(loss.item())
                # _, predicted = torch.max(outputs, 1)
                # correct = (predicted == labels).sum().item()
                # accuracy = correct / len(labels)
                # valid_accuracy_list.append(accuracy)
            logger.info('valid loss: %f', np.mean(valid_loss_list))
            logger.info('valid accuracy: %f', np.mean(valid_accuracy_list))
            writer.add_scalar('valid_loss', np.mean(valid_loss_list), epoch)
            writer.add_scalar('valid_accuracy', np.mean(valid_accuracy_list), epoch)
        
    logger.info('training finished')
        
