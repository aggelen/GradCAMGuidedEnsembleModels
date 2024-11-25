#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:38:27 2024

@author: gelenag
"""

from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torch.utils.data import Dataset, DataLoader

import numpy as np

default_data_path = "../dataset/data"
def get_dataset_stats():
    train_data = CIFAR100(root=default_data_path, train=True, download=True)
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255
    mean=mean.tolist()
    std=std.tolist()
    return mean, std

def CIFAR100Loaders(batch_size=64):
    mean, std = get_dataset_stats()
    
    transform_train = tt.Compose([tt.RandomHorizontalFlip(),  
                                  tt.RandomRotation(15),     
                                  tt.RandomCrop(32, padding=4),
                                  tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Renk değişiklikleri
                                  tt.ToTensor(),
                                  tt.Normalize(mean,std,inplace=True)])
    
    transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean,std)])
    
    trainset = CIFAR100(default_data_path,
                        train=True,
                        download=True,
                        transform=transform_train)
    
    testset = CIFAR100(default_data_path,
                       train=False,
                       download=True,
                       transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, pin_memory=True, num_workers=2)
    
    return trainloader, testloader