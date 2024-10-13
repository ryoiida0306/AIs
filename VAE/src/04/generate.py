from datetime import datetime
import logging

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import DataSet, get_data_loader
import os

from torch.utils.tensorboard import SummaryWriter  # Add this line
from torch.optim import Adam


logger = logging.getLogger(__name__)

def runtime_generate(model, logs_dir, checkpoint_dir, loading) :
    load = loading[0]
    if load == True:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{loading[1]}.pth')
        model.load_state_dict(torch.load(checkpoint_path))
    
    # writer = SummaryWriter(log_dir = logs_dir)

    run = True
    logger.info('generate started')
    model.eval()
    while run :
        user_input = input('Enter the mu to generate image (2 dimentions): ')
        mu = torch.tensor([float(i) for i in user_input.split()])
        # user_input = input('Enter the log_var to generate image (2 dimentions): ')
        # log_var = torch.tensor([float(i) for i in user_input.split()])
        log_var = torch.tensor([0, 0])
        with torch.no_grad() :
            outputs = model.generate(mu, log_var)
            plt.imshow(outputs[0].reshape(28, 28), cmap='gray')
            plt.show()
        user_input = input('Do you want to generate another image? (y/n): ')
        if user_input == 'n' :
            run = False
    logger.info('generate finished')

if __name__ == '__main__' :
   
    # 現在の作業ディレクトリのパスを取得
    current_working_directory = os.getcwd()

    # パスから現在のディレクトリ名を取得
    current_folder_name = os.path.basename(current_working_directory)

    save_dir = '../../result/' + current_folder_name
    generate_dir = os.path.join(save_dir, 'gen')
    logs_dir = os.path.join(generate_dir, 'logs')
    checkpoint_dir = os.path.join(save_dir, 'models')
    os.makedirs(logs_dir, exist_ok=True)
    input_path = 'F:\\mnist\\mnist_X.pkl'
    label_path = 'F:\\mnist\\mnist_Y.pkl'
    # load = True
    load_epoch = 1500
    loading = [True, load_epoch]

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(filename=os.path.join(logs_dir, f'train_{current_time}.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    # train_loader, valid_loader, test_loader = get_data_loader(input_path, label_path, 64)

    model = Model(28*28, 256, 2)
    optimizer = Adam(model.parameters(), lr=0.005)

    runtime_generate(model, logs_dir, checkpoint_dir, loading)