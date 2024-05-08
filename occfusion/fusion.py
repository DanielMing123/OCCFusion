import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from mmengine.runner.amp import autocast

class DynamicFusion3D(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()  
        self.refine = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels // 2, out_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.refine(x)
        # return x
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
        

class DynamicFusion2D(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()  
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels // 2, out_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.refine(x)
        # return x
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
        