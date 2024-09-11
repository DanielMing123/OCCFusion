import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet.models.utils.misc import multi_apply

# Fully Connected Network
class FCN(nn.Module):
    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self,x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t,-1))
        x = F.relu(self.bn(x))
        return x.view(kk,t,-1)

# Voxel Feature Encoding layer
class VFE(nn.Module):
    def __init__(self, cin, cout, T):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin,self.units)
        self.T = T

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        #locally aggregated feature
        laf = torch.max(pwf,1)[0].unsqueeze(1).repeat(1,self.T,1)
        # point-wise concat feature
        pwcf = torch.cat((pwf,laf),dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


@MODELS.register_module()
class SVFE(BaseModule):

    def __init__(self, 
                 num_pts,
                 input_dim,
                 grid_size):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(input_dim,16,num_pts)
        self.vfe_2 = VFE(16,32,num_pts)
        self.fcn = FCN(32,64)
        self.grid_size = grid_size
        self.bn = nn.BatchNorm1d(input_dim)
        
    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1] # dim is the feature channel
        dense_feature = torch.zeros(dim, 
                                    self.grid_size[2], 
                                    self.grid_size[0], 
                                    self.grid_size[1],
                                    dtype=sparse_features.dtype,
                                    device=sparse_features.device) # [C,Z,X,Y] or [C,D,H,W]
        dense_feature[:, coords[:,0], coords[:,1], coords[:,2]]= sparse_features.transpose(0,1).contiguous()
        return dense_feature
    
    def _single_forward(self,sparse_feat, voxel_coord):
        mask = torch.ne(torch.max(sparse_feat,2)[0], 0)
        sparse_feat = self.vfe_1(sparse_feat, mask)
        sparse_feat = self.vfe_2(sparse_feat, mask)
        sparse_feat = self.fcn(sparse_feat)
        # element-wise max pooling
        sparse_feat = torch.max(sparse_feat,1)[0]
        dense_3d_volume = self.voxel_indexing(sparse_feat, voxel_coord)
        return dense_3d_volume
    
    def forward(self, sparse_feats, voxel_coords):
        batch_res = []
        for sparse_feat, voxel_coord in zip(sparse_feats,voxel_coords):
            sparse_feat = self.bn(sparse_feat.transpose(1,2)).transpose(1,2).contiguous()
            batch_res.append(self._single_forward(sparse_feat, voxel_coord))
        batch_res = torch.stack(batch_res,dim=0)

        return batch_res.permute(0,1,3,4,2)