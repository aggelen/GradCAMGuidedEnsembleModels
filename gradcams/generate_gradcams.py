#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:10:48 2024

@author: gelenag
"""
import sys
sys.path.append('../models')
sys.path.append('../dataset')

from Models import ResNet18, DilatedResNet18, SEResNet18, MobileNetV3L, GoogleNet
from Dataset import CIFAR100Loaders
from Training import train_model_with_early_stopping

from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.close("all")

# Load models
models = {
    'resnet18': ResNet18(),
    'dilated_resnet18': DilatedResNet18(),
    'se_resnet18': SEResNet18(),
    'mobilenet_v3l': MobileNetV3L(),
    'googlenet': GoogleNet()
}

cps = {
    'resnet18': '../chekpoints/resnet18_best_model_checkpoint.pth',
    'dilated_resnet18': '../chekpoints/dilated_resnet18_best_model_checkpoint.pth',
    'se_resnet18': '../chekpoints/se_resnet18_best_model_checkpoint.pth',
    'mobilenet_v3l': '../chekpoints/obilenet_v3l_best_model_checkpoint.pth',
    'googlenet': '../chekpoints/googlenet_best_model_checkpoint.pth'
}

batch_size = 1

#%% Dataset
_, _, trainset, testset = CIFAR100Loaders(batch_size, return_datasets=True)

train_loader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=False)
upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
grad_cams = {m: [] for m in models.keys()}
for model_name, model in models.items():
    model.to(device)
    checkpoint = torch.load(cps[model_name], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if model_name in ['resnet18', 'dilated_resnet18', 'se_resnet18']:
        target_layers = [model.resnet18.layer4[-1]]
    elif model_name == 'mobilenet_v3l':
        target_layers = [model.net.features[-1]]
    elif model_name == 'googlenet':
        target_layers = [model.googlenet.inception5b]
    correct_grad_cams_i = {i: [] for i in range(100)}
    incorrect_grad_cams_i = {i: [] for i in range(100)}
    
    no_correct = 0
    no_incorrect = 0
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for i, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # rgb_image = tt.functional.to_pil_image(denormalize(data, mean, std)[0])
        
            #%%
            pred = model(data.to(device))
            #%%                
            targets = [ClassifierOutputTarget(labels.item())]
            grayscale_cam = cam(input_tensor=data, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            model_outputs = cam.outputs
            grad_cams[model_name].append(grayscale_cam)
            
with open('grad_cams_train_all.pkl', 'wb') as fp:
    pickle.dump(grad_cams, fp)
