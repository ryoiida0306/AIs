import torch
import torch.nn as nn
import numpy as np
# import torch.optim as optim
# import torch.nn.functional as F
from torchvision import transforms

class DataSet(torch.utils.data.Dataset) :

    def __init__(self, data, labels) :
        self.data = np.array(data, dtype=np.float32)
        # self.data = self.data.astype(np.uint8)
        self.data = self.data.reshape(len(data), 1, 28, 28)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.labels = labels
        self.data = self.data[:1000, :, :, :]
        self.labels = self.labels[:1000]

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        image = self.data[idx]
        # print(image.shape)
        image = self.transform(image)
        # print(image.shape)
        label = self.labels[idx]
        return image, label

def get_data_loader(input_path, label_path, batch_size, split_ratio = (0.8, 0.1, 0.1)) :

    with open(input_path, 'rb') as f :
        input_data = np.load(f, encoding='bytes', allow_pickle=True)
    
    with open(label_path, 'rb') as f :
        label_data = np.load(f, encoding='bytes', allow_pickle=True)
    ds = DataSet(input_data, label_data)
    # train = None
    # valid = None
    # test = None

    train_size = int(split_ratio[0] * len(ds))
    valid_size = int(split_ratio[1] * len(ds))
    test_size = len(ds) - train_size - valid_size

    train = torch.utils.data.Subset(ds, range(0, train_size))
    valid = torch.utils.data.Subset(ds, range(train_size, train_size + valid_size))
    test = torch.utils.data.Subset(ds, range(train_size + valid_size, len(ds)))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader