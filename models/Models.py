#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:27:59 2024

@author: gelenag
"""

import torch
import torch.nn as nn
from torchvision import models
from Utils import dilated_resnet18, se_resnet18, resnet18
# from torchvision.models import ResNet

#%% Resnet 18
class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet18 = resnet18()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.resnet18(x)
    
    def encoder_forward(self, x):
        x = self.upsample(x)
        encoder = nn.Sequential(*list(self.resnet18.children())[:-1]) 
        x = encoder(x)
        return x.view(x.shape[0], -1) 

#%% Dilated Resnet 18
class DilatedResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet18 = dilated_resnet18()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.resnet18(x)
    
    def encoder_forward(self, x):
        x = self.upsample(x)
        encoder = nn.Sequential(*list(self.resnet18.children())[:-1]) 
        x = encoder(x)
        return x.view(x.shape[0], -1) 

#%% SE Resnet 18
class SEResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet18 = se_resnet18()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(256, 100)
        )
        
    def encoder_forward(self, x):
        x = self.upsample(x)
        encoder = nn.Sequential(*list(self.resnet18.children())[:-1]) 
        x = encoder(x)
        return x.view(x.shape[0], -1) 

    def forward(self, x):
        x = self.upsample(x)
        return self.resnet18(x)


#%% MobileNetv3L
class MobileNetV3L(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.net = models.mobilenet_v3_large()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.net.classifier[0] = torch.nn.Linear(960, 256)
        self.net.classifier[3] = torch.nn.Linear(256, 100)

    def forward(self, x):
        x = self.upsample(x)
        return self.net(x)
    
    def encoder_forward(self, x):
        x = self.upsample(x)
        encoder = nn.Sequential(*list(self.net.children())[:-1]) 
        x = encoder(x)
        return x.view(x.shape[0], -1) 


#%% GoogLeNet
class GoogleNet(nn.Module):
    def __init__(self, num_classes=100):
        super(GoogleNet, self).__init__()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.googlenet = models.googlenet(init_weights=False, aux_logits=False)
        self.googlenet.fc = nn.Sequential(nn.Linear(self.googlenet.fc.in_features, 256),
                                          nn.Dropout(0.25),
                                          nn.LeakyReLU(),
                                          nn.Linear(256, 100)
                                        )
    def forward(self, x):
        x = self.upsample(x)
        return self.googlenet(x)
    
    def encoder_forward(self, x):
        x = self.upsample(x)
        encoder = nn.Sequential(*list(self.googlenet.children())[:-1]) 
        x = encoder(x)
        return x.view(x.shape[0], -1) 

