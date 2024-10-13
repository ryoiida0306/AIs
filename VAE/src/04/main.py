import os
import logging
import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import DataSet, get_data_loader
from train import train
from test import test
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # Add this line
from torch.optim import Adam



# 現在の作業ディレクトリのパスを取得
current_working_directory = os.getcwd()

# パスから現在のディレクトリ名を取得
current_folder_name = os.path.basename(current_working_directory)

save_dir = '../../result/' + current_folder_name
logs_dir = os.path.join(save_dir, 'logs')
model_dir = os.path.join(save_dir, 'models')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
input_path = 'F:\\mnist\\mnist_X.pkl'
label_path = 'F:\\mnist\\mnist_Y.pkl'
epochs = 2000
load = False
load_epoch = 0
loading = [load, load_epoch]

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(save_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logs_dir, f'train_{current_time}.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

train_loader, valid_loader, test_loader = get_data_loader(input_path, label_path, 64)

model = Model(28*28, 256, 2)
optimizer = Adam(model.parameters(), lr=0.005)

train(model, train_loader, valid_loader, epochs, optimizer, model_dir, loading)
test(model, test_loader)