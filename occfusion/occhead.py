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
        self.mlp_occs = nn.ModuleList()
        for i in range(len(self.channels)):
            mlp_occ = nn.Sequential(
                nn.Linear(self.channels[i], self.channels[i]),
                nn.ReLU(),
                nn.Linear(self.channels[i], self.channels[i]),
                nn.ReLU(),
                nn.Linear(self.channels[i], self.num_cls)
            )
            self.mlp_occs.append(mlp_occ)
    
    @autocast('cuda',torch.float32)
    def forward(self, xyz_volumes):        
        if self.training:
            logits = []
            for mlp_occ,xyz_volume in zip(self.mlp_occs, xyz_volumes):
                logit = mlp_occ(xyz_volume.permute(0,2,3,4,1))
                logits.append(logit)
            # logits = self.mlp_occs[0](xyz_volumes.permute(0,2,3,4,1))
            return logits
        else:
            logits_lvl0 = self.mlp_occs[0](xyz_volumes[0].permute(0,2,3,4,1))
            return logits_lvl0
            # logits = self.mlp_occs[0](xyz_volumes.permute(0,2,3,4,1))
            # return logits
                
        
