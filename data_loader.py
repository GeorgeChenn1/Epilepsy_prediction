import numpy as np
from torchvision import datasets, transforms
import torch
import os
from scipy.io import loadmat
from EEGDataset import EEGDataset

def load_training(root_path, data, batch_size, kwargs):
    # transform = transforms.Compose(
    #     [transforms.Resize([256, 256]),
    #      transforms.RandomCrop(224),
    #      transforms.RandomHorizontalFlip(),
    #      transforms.ToTensor()])
    # data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    # f0 = loadmat("../seizure5times/data/eeg1_ds.mat")
    # data = f0['x']
    # [((3726, 64, 19), (3726,)), ((3726, 64, 19), (3726,)),((3726, 64, 19), (3726,)),((3726, 64, 19), (3726,))]
    arr_x = []
    arr_y = []
    for i in range(len(data)):
        arr_x.append(data[i][0]) # (3726, 64, 19)
        arr_y.append(data[i][1]) # (3726,)

    x = np.concatenate(arr_x, axis=0) # [(3726, 64, 19),(3726, 64, 19),(3726, 64, 19)]
    y = np.concatenate(arr_y, axis=0)  # [(3726,),(3726,),(3726,)]
    x = np.transpose(x[..., np.newaxis], (0,3,1,2))
    dataset = EEGDataset((x,y))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, data, batch_size, kwargs, drop_last):
    # transform = transforms.Compose(
    #     [transforms.Resize([224, 224]),
    #      transforms.ToTensor()])
    # data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    x, y = data
    #test_loader = torch.utils.data.DataLoader(tmp, batch_size=1, shuffle=True, drop_last=drop_last, **kwargs)
    #print(type(test_loader))
    #print(test_loader)
    #input('')
    x = np.transpose(x[..., np.newaxis], (0,3,1,2))
    dataset = EEGDataset((x,y))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
    return test_loader