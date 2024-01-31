import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from algorithms import OBC


def get_custom_datasets(path, suffix=''):
    x_train = torch.load(os.path.join(path, f'features_train{suffix}.pt'))
    y_train = torch.load(os.path.join(path, f'targets_train{suffix}.pt'))
    x_val = torch.load(os.path.join(path, f'features_val{suffix}.pt'))
    y_val = torch.load(os.path.join(path, f'targets_val{suffix}.pt'))
    data_train = CustomDataset(data=x_train, targets=y_train)
    data_val = CustomDataset(data=x_val, targets=y_val)
    return data_train, data_val


def get_rn50x16openai_datasets(path):
    return get_custom_datasets(path)