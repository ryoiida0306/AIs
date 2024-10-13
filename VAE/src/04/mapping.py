import logging
import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import DataSet, get_data_loader
from plot import save_tensors_as_pdf
from datetime import datetime
import os
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter  # Add this line
from torch.optim import Adam
from torch.utils import tensorboard
import matplotlib.pyplot as plt




def mapping(model, test_loader, logs_dir, checkpoint_dir, map_dir, loading) :
    load = loading[0]
    if load == True:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{loading[1]}.pth')
        model.load_state_dict(torch.load(checkpoint_path))

    logger.info('mapping started')
    writer = SummaryWriter(log_dir = logs_dir)

    # logging_interval = 10

    qz_labels_map = torch.tensor([])
    all_mu = torch.tensor([])
    all_log_var = torch.tensor([])
    all_labels = torch.tensor([])
    
    model.eval()
    for batch in test_loader :
        inputs = batch[0]
        labels = batch[1]
        # target = inputs
        outputs, mu, log_var = model(inputs)
        all_mu = torch.cat((all_mu, mu), dim=0)
        all_log_var = torch.cat((all_log_var, log_var), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        # qz_labels_map.append([[mu, log_var], labels])
    print(all_mu.shape)
    print(all_log_var.shape)
    print(all_labels.shape)
    
    qz_labels_map = [all_mu, all_log_var, all_labels]
    pkl.dump(qz_labels_map, open(os.path.join(map_dir, 'qz_labels_map.pkl'), 'wb'))
    logger.info('mapping finished')
        
if __name__ == '__main__' :
   
    # 現在の作業ディレクトリのパスを取得
    current_working_directory = os.getcwd()

    # パスから現在のディレクトリ名を取得
    current_folder_name = os.path.basename(current_working_directory)

    save_dir = '../../result/' + current_folder_name
    mapping_dir = os.path.join(save_dir, 'mapping')
    logs_dir = os.path.join(mapping_dir, 'logs')
    map_dir = os.path.join(mapping_dir, 'map')
    checkpoint_dir = os.path.join(save_dir, 'models')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    input_path = 'F:\\mnist\\mnist_X.pkl'
    label_path = 'F:\\mnist\\mnist_Y.pkl'
    # load = True
    load_epoch = 1500
    loading = [True, load_epoch]

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(filename=os.path.join(logs_dir, f'train_{current_time}.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    train_loader, valid_loader, test_loader = get_data_loader(input_path, label_path, 64)

    model = Model(28*28, 256, 2)
    optimizer = Adam(model.parameters(), lr=0.005)

    mapping(model, test_loader, logs_dir, checkpoint_dir, map_dir, loading)
