import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import pickle
from torch.utils.data import DataLoader
import scipy.spatial as sp

class SimpleDataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx])

    def __len__(self):
        return len(self.X)
    
    def dataset(self):
        return self.X


def get_permuted_mnist_train_data(task_id, batch_size = 256 , rndseed=42):

    rng_permute = np.random.RandomState(rndseed*100+task_id)
    idx_permute = torch.from_numpy(rng_permute.permutation(784))
    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                  torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) )])

    #image datasets
    train_dataset = torchvision.datasets.MNIST('dataset/', 
                                               train=True, 
                                               download=True,
                                               transform=image_transform)
    #data loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True)

    return train_dataloader
    

def get_permuted_mnist_test_data(task_id,  batch_size = 1024, rndseed=42):

    rng_permute = np.random.RandomState(rndseed*100+task_id)
    idx_permute = torch.from_numpy(rng_permute.permutation(784))
    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                  torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) )])

    test_dataset = torchvision.datasets.MNIST('dataset/', 
                                              train=False, 
                                              download=True,
                                              transform=image_transform)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=4,
                                              shuffle=True)
    
    return test_dataloader


def get_rotated_mnist_train_data(task_id, batch_size = 256 , rndseed=42):

    image_transform = torchvision.transforms.Compose((Rotation(rndseed*100+task_id), torchvision.transforms.ToTensor()))

    #image datasets
    train_dataset = torchvision.datasets.MNIST('dataset/', 
                                               train=True, 
                                               download=True,
                                               transform=image_transform)
    #data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True)
    return train_loader

def get_rotated_mnist_test_data(task_id,  batch_size = 1024, rndseed=42):

    image_transform = torchvision.transforms.Compose((Rotation(rndseed*100+task_id), torchvision.transforms.ToTensor()))

    test_dataset = torchvision.datasets.MNIST('dataset/', 
                                              train=False, 
                                              download=True,
                                              transform=image_transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=4,
                                              shuffle=True)
    return test_loader

class Rotation(object):

    def __init__(self, seed, deg_min: int = 0, deg_max: int = 180) -> None:

        self.deg_min = deg_min
        self.deg_max = deg_max
        rng_permute = np.random.seed(seed)
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)
        print(f"ROTATION: {self.degrees}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return torchvision.transforms.functional.rotate(x, self.degrees)

