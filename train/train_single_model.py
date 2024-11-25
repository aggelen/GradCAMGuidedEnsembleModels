#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:50:35 2024

@author: gelenag
"""
import sys
sys.path.append('../models')
sys.path.append('../dataset')

from Models import ResNet18, DilatedResNet18, SEResNet18, MobileNetV3L, GoogleNet
from Dataset import CIFAR100Loaders
from Training import train_model_with_early_stopping

#%% Configuration
models = {'resnet18': ResNet18(),
          'dilated_resnet18': DilatedResNet18(),
          'se_resnet18': SEResNet18(),
          'mobilenet_v3l': MobileNetV3L(),
          'googlenet': GoogleNet()
          }

batch_size = 64

#%% Dataset
trainloader, testloader = CIFAR100Loaders(batch_size)

for model_name, model in models.items():
    print(f'{model_name} training ...')
    trained_model = train_model_with_early_stopping(model, trainloader, testloader, batch_size, model_name, num_epochs=200, patience=5)
    print(f'{model_name} trained ...')