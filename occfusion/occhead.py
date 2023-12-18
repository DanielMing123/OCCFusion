import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner.amp import autocast

@MODELS.register_module()
class OccHead(BaseModule):
    def __init__(self,
                 channels,
                 num_classes):
        super(OccHead, self).__init__()
        self.channels = channels
        self.num_cls = num_classes        
        self.mlp_occ_lvl0 = nn.Sequential(
            nn.Linear(self.channels[0], self.channels[0]),
            nn.ReLU(),
            nn.Linear(self.channels[0], self.channels[0]),
            nn.ReLU(),
            nn.Linear(self.channels[0], self.num_cls)
            )
        self.mlp_occ_lvl1 = nn.Sequential(
            nn.Linear(self.channels[1], self.channels[1]),
            nn.ReLU(),
            nn.Linear(self.channels[1], self.channels[1]),
            nn.ReLU(),
            nn.Linear(self.channels[1], self.num_cls)
            )
        self.mlp_occ_lvl2 = nn.Sequential(
            nn.Linear(self.channels[2], self.channels[2]),
            nn.ReLU(),
            nn.Linear(self.channels[2], self.channels[2]),
            nn.ReLU(),
            nn.Linear(self.channels[2], self.num_cls)
            )
        self.mlp_occ_lvl3 = nn.Sequential(
            nn.Linear(self.channels[3], self.channels[3]),
            nn.ReLU(),
            nn.Linear(self.channels[3], self.channels[3]),
            nn.ReLU(),
            nn.Linear(self.channels[3], self.num_cls)
            )
    
    @autocast('cuda',torch.float32)
    def forward(self,
                xyz_feat_lvl0,
                xyz_feat_lvl1,
                xyz_feat_lvl2,
                xyz_feat_lvl3):        
        if self.training:
            logits_lvl0 = self.mlp_occ_lvl0(xyz_feat_lvl0.permute(0,2,3,4,1))
            logits_lvl1 = self.mlp_occ_lvl1(xyz_feat_lvl1.permute(0,2,3,4,1))
            logits_lvl2 = self.mlp_occ_lvl2(xyz_feat_lvl2.permute(0,2,3,4,1))
            logits_lvl3 = self.mlp_occ_lvl3(xyz_feat_lvl3.permute(0,2,3,4,1))
            return logits_lvl0, logits_lvl1, logits_lvl2, logits_lvl3
        else:
            logits_lvl0 = self.mlp_occ_lvl0(xyz_feat_lvl0.permute(0,2,3,4,1))
            return logits_lvl0
                
        
