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

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

batch_size = 64

#%% Dataset
trainloader, testloader, trainset, testset = CIFAR100Loaders(batch_size, return_datasets=True)

class_names = trainset.classes

def plot_confusion_matrix(cm, ax, model_name, norm):
    cax = ax.imshow(cm, cmap='inferno', norm=norm)
    ax.set_title(model_name)
    ax.set_xticks(np.arange(0, len(class_names), 20))
    ax.set_yticks(np.arange(0, len(class_names), 20))
    ax.set_xticklabels(class_names[::20], fontsize=6)  # Daha küçük font
    ax.set_yticklabels(class_names[::20], fontsize=6)  # Daha küçük font
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)


fig, ax = plt.subplots(figsize=(25, 5), dpi=300, layout="constrained")  # 5 modeli yerleştirebileceğimiz tek büyük grafik
log_norm = LogNorm()

n_models = len(models)
cm_list = []
for i, (model_name, model) in enumerate(models.items()):
    model.to(device)
    
    checkpoint = torch.load(cps[model_name], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_list.append(cm)
    if i < 4:
        cm_list.append(np.zeros((100,5)))

#%%
combined_cm = np.concatenate(cm_list, axis=1)

cax = ax.imshow(combined_cm, cmap='inferno', norm=log_norm)

ax.set_xticks([50, 150+5, 250+10, 350+15, 450+20])
ax.set_xticklabels([model_name for model_name in models.keys()], fontsize=10)
ax.set_yticks(np.arange(0, len(class_names), 20))
ax.set_yticklabels(class_names[::20], fontsize=6)

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('True', fontsize=12)

cbax = ax.inset_axes([0, -0.2, 0.7, 0.05], transform=ax.transAxes)
fig.colorbar(cax, ax=ax, cax=cbax, orientation='horizontal')

# Grafiği kaydedip göster
# plt.tight_layout(pad=3, w_pad=0.0, h_pad=3)
# plt.tight_layout(pad=0., w_pad=0.3, h_pad=1.5)
# plt.savefig('Figs/combined_confusion_matrix_A4.png', dpi=300)  # A4 boyutunda ve yüksek çözünürlükle kaydet
plt.show()
