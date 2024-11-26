#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:12:15 2024

@author: gelenag
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.close("all")

cps_re18 = torch.load("../chekpoints/resnet18_best_model_checkpoint.pth", map_location='cpu')
cps_se_re18 = torch.load("../chekpoints/se_resnet18_best_model_checkpoint.pth", map_location='cpu')
cps_mobilenetv3l = torch.load("../chekpoints/mobilenet_v3l_best_model_checkpoint.pth", map_location='cpu')
cps_googlenet = torch.load("../chekpoints/googlenet_best_model_checkpoint.pth", map_location='cpu')
cps_dilated = torch.load("../chekpoints/dilated_resnet18_best_model_checkpoint.pth", map_location='cpu')

train_acc_re18 = cps_re18['train_accuracy_history']
test_acc_re18 = cps_re18['test_accuracy_history']

train_acc_se_re18 = cps_se_re18['train_accuracy_history']
test_acc_se_re18 = cps_se_re18['test_accuracy_history']

train_acc_mobilenetv3l = cps_mobilenetv3l['train_accuracy_history']
test_acc_mobilenetv3l = cps_mobilenetv3l['test_accuracy_history']

train_acc_googlenet = cps_googlenet['train_accuracy_history']
test_acc_googlenet = cps_googlenet['test_accuracy_history']

train_acc_dilated = cps_dilated['train_accuracy_history']
test_acc_dilated = cps_dilated['test_accuracy_history']

plt.figure(figsize=(11, 8))

epochs_re18 = range(1, len(train_acc_re18) + 1)
plt.plot(epochs_re18, train_acc_re18, 'r-', alpha=0.8, label="ResNet-18 Train", linewidth=2)
plt.plot(epochs_re18, test_acc_re18, 'r--', alpha=0.8, label="ResNet-18 Test", linewidth=2)
plt.scatter(len(train_acc_re18), train_acc_re18[-1], color='r', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti
plt.scatter(len(test_acc_re18), test_acc_re18[-1], color='r', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti

epochs_se_re18 = range(1, len(train_acc_se_re18) + 1)
plt.plot(epochs_se_re18, train_acc_se_re18, 'b-', alpha=0.7, label="SE-ResNet-18 Train", linewidth=2)
plt.plot(epochs_se_re18, test_acc_se_re18, 'b--', alpha=0.7, label="SE-ResNet-18 Test", linewidth=2)
plt.scatter(len(train_acc_se_re18), train_acc_se_re18[-1], color='b', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti
plt.scatter(len(test_acc_se_re18), test_acc_se_re18[-1], color='b', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti

epochs_mobilenetv3l = range(1, len(train_acc_mobilenetv3l) + 1)
plt.plot(epochs_mobilenetv3l, train_acc_mobilenetv3l, 'g-', alpha=0.7, label="MobileNetV3-Large Train", linewidth=2)
plt.plot(epochs_mobilenetv3l, test_acc_mobilenetv3l, 'g--', alpha=0.7, label="MobileNetV3-Large Test", linewidth=2)
plt.scatter(len(train_acc_mobilenetv3l), train_acc_mobilenetv3l[-1], color='g', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti
plt.scatter(len(test_acc_mobilenetv3l), test_acc_mobilenetv3l[-1], color='g', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti

epochs_googlenet = range(1, len(train_acc_googlenet) + 1)
plt.plot(epochs_googlenet, train_acc_googlenet, 'orange', linestyle='-', alpha=0.6, label="GoogLeNet Train", linewidth=2)
plt.plot(epochs_googlenet, test_acc_googlenet, 'orange', linestyle='--', alpha=0.6, label="GoogLeNet Test", linewidth=2)
plt.scatter(len(train_acc_googlenet), train_acc_googlenet[-1], color='orange', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti
plt.scatter(len(test_acc_googlenet), test_acc_googlenet[-1], color='orange', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti

epochs_dilated = range(1, len(train_acc_dilated) + 1)
plt.plot(epochs_dilated, train_acc_dilated, 'purple', linestyle='-', alpha=0.6, label="Dilated ResNet-18 Train", linewidth=2)
plt.plot(epochs_dilated, test_acc_dilated, 'purple', linestyle='--', alpha=0.6, label="Dilated ResNet-18 Test", linewidth=2)
plt.scatter(len(train_acc_dilated), train_acc_dilated[-1], color='purple', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti
plt.scatter(len(test_acc_dilated), test_acc_dilated[-1], color='purple', zorder=5, s=80, edgecolor='black')  # Bitiş noktası işareti

plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.minorticks_on()  # Küçük grid çizgilerini aktif hale getir
plt.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, alpha=0.7)  
plt.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5, alpha=0.5)  

plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))

plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()

plt.savefig('Figs/model_accuracy_comparison_A4.png', dpi=300)

plt.show()
