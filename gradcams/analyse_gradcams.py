#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:10:48 2024

@author: gelenag
"""

import numpy as np
import cv2
from PIL import Image
from Utils import cifar100_labels

import pickle
import os
import matplotlib.pyplot as plt
plt.close("all")

with open('grad_cams.pkl', 'rb') as fp:
    correct_grad_cams_all = pickle.load(fp)
    incorrect_grad_cams_all = pickle.load(fp)

#%%
# apple_list = [9, 113, 226, 377, 469, 484, 614, 623, 715, 655, 779, 1014, 1027, 1136, 1304, 1308]
label_id = 77
models = ['resnet18', 'dilated_resnet18', 'se_resnet18', 'mobilenet_v3l', 'googlenet']
model_names = ['Resnet 18', 'Dilated Resnet18', 'SE Resnet18', 'Mobilenet v3 L', 'GoogLeNet']
#%%
sample_idx = []
for m in models:
    sample_idx_i = []
    for k in range(20):
        sample_idx_i.append(correct_grad_cams_all[m][label_id][k][2])
    sample_idx.append(sample_idx_i)

sample_idx = np.array(sample_idx)
common_idx = []
for e in sample_idx[0]:
    if np.sum(sample_idx == e,1).sum() == 5:
        common_idx.append(e)
    
    if len(common_idx) == 9:
        break

os.makedirs(f'Gradcams/{cifar100_labels[label_id]}', exist_ok=True)

for mid, m in enumerate(models):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), layout="constrained")
    plt.suptitle(f"{model_names[mid]} GradCams for class: {cifar100_labels[label_id]}", fontsize="16")
    for i, ax in enumerate(axes.flat):
        for kk in range(len(correct_grad_cams_all[m][label_id])): 
            if correct_grad_cams_all[m][label_id][kk][2] == common_idx[i]:
                ax.imshow(correct_grad_cams_all[m][label_id][kk][0]) 
                # print(correct_grad_cams_all[m][label_id][i][2])
                ax.axis('off')
                break
    
    plt.savefig(f'Gradcams/{cifar100_labels[label_id]}/{model_names[mid]}.png', dpi=300)
    plt.close()