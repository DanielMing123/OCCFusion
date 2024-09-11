from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from focal_loss.focal_loss import FocalLoss
from mmdet3d.models.losses import LovaszLoss
from .loss import geo_scal_loss, sem_scal_loss


@MODELS.register_module()
class OccFusion(Base3DSegmentor):
    def __init__(self,
                 use_occ3d,
                 use_lidar,
                 use_radar,
                 data_preprocessor,
                 backbone,
                 neck,
                 view_transformer,
                 svfe_lidar,
                 svfe_radar,
                 occ_head):
        super().__init__(data_preprocessor=data_preprocessor)
        self.occ3d = use_occ3d
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.img_backbone = MODELS.build(backbone)
        self.img_neck = MODELS.build(neck)
        if self.use_lidar:
            self.svfe_lidar = MODELS.build(svfe_lidar)
        if self.use_radar:
            self.svfe_radar = MODELS.build(svfe_radar)
        self.view_transformer = MODELS.build(view_transformer)
        self.occ_head = MODELS.build(occ_head)
        self.loss_fl = FocalLoss(gamma=2,ignore_index=255) # 0: noise label weights=
        self.loss_lovasz = LovaszLoss(loss_type='multi_class',
                                      per_sample=False,
                                      reduction='none')

    def multiscale_supervision(self, gt_occ, ratio, gt_shape):
        gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]]).to(gt_occ[0].device).type(torch.long) 
        for i in range(gt.shape[0]):
            coords_x = gt_occ[i][:, 0].to(torch.float) // ratio[0]
            coords_y = gt_occ[i][:, 1].to(torch.float) // ratio[1]
            coords_z = gt_occ[i][:, 2].to(torch.float) // ratio[2]
            coords_x = coords_x.to(torch.long)
            coords_y = coords_y.to(torch.long)
            coords_z = coords_z.to(torch.long)
            coords = torch.stack([coords_x,coords_y,coords_z],dim=1)
            gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]
        
        return gt
    
    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped    
    
    def _forward(self, batch_inputs,batch_data_samples): # img, img_metas, sparse_voxel, voxel_coords, batch_data_samples
        """Forward training function."""
        imgs = batch_inputs['imgs']
        img_metas = []
        for data_sample in batch_data_samples:
            if not self.occ3d:
                img_meta = dict(lidar2img=data_sample.lidar2img,
                                img_shape=torch.Tensor([imgs.shape[-2],imgs.shape[-1]])
                                )
            else:
                img_meta = dict(lidar2img=data_sample.ego2img,
                                img_shape=torch.Tensor([imgs.shape[-2],imgs.shape[-1]])
                                )
            img_metas.append(img_meta)
        
        img_feats = self.extract_feat(imgs)
        if self.use_lidar:
            lidar_xyz_feat = self.svfe_lidar(batch_inputs['lidar_voxel_feats'], batch_inputs['lidar_voxel_coords']) # [B, C, X, Y, Z]
        if self.use_radar:
            radar_xyz_feat = self.svfe_radar(batch_inputs['radar_voxel_feats'], batch_inputs['radar_voxel_coords'])
        if (not self.use_lidar) and (not self.use_radar):
            xyz_volumes = self.view_transformer(img_feats, img_metas) # [B, C, X, Y, Z]
        elif self.use_lidar and (not self.use_radar):
            xyz_volumes = self.view_transformer.forward_two(img_feats, img_metas, lidar_xyz_feat) #  lidar_xyz_feat
        elif (not self.use_lidar) and self.use_radar:
            xyz_volumes = self.view_transformer.forward_two(img_feats, img_metas, radar_xyz_feat)
        elif self.use_lidar and self.use_radar:
            xyz_volumes = self.view_transformer.forward_three(img_feats, img_metas, lidar_xyz_feat, radar_xyz_feat)
        return self.occ_head(xyz_volumes) # fused_xyz_feat
             
    def loss(self,batch_inputs, batch_data_samples):
        vox_logits_lvl0, vox_logits_lvl1,vox_logits_lvl2,vox_logits_lvl3 = self._forward(batch_inputs,batch_data_samples)
        B,X,Y,Z,Cls = vox_logits_lvl0.shape
        if X==256:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[2,2,2],np.array([len(batch_data_samples),128,128,16],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[4,4,4],np.array([len(batch_data_samples),64,64,8],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[8,8,8],np.array([len(batch_data_samples),32,32,4],dtype=np.int32))
        elif self.occ3d:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_3d_masked'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['occ_3d'],[2,2,2],np.array([len(batch_data_samples),100,100,8],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['occ_3d'],[4,4,4],np.array([len(batch_data_samples),50,50,4],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['occ_3d'],[8,8,8],np.array([len(batch_data_samples),25,25,2],dtype=np.int32))
        else:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[2,2,2],np.array([len(batch_data_samples),100,100,8],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[4,4,4],np.array([len(batch_data_samples),50,50,4],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[8,8,8],np.array([len(batch_data_samples),25,25,2],dtype=np.int32))
            
        vox_fl_predict_lvl0 = vox_logits_lvl0.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl0 = vox_logits_lvl0.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl0 = vox_logits_lvl0.reshape(-1,Cls)
        vox_lovasz_label_lvl0 = voxels_lvl0.reshape(-1)
        
        vox_fl_predict_lvl1 = vox_logits_lvl1.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl1 = voxels_lvl1.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl1 = vox_logits_lvl1.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl1 = vox_logits_lvl1.reshape(-1,Cls)
        vox_lovasz_label_lvl1 = voxels_lvl1.reshape(-1)
        
        vox_fl_predict_lvl2 = vox_logits_lvl2.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl2 = voxels_lvl2.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl2 = vox_logits_lvl2.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl2 = vox_logits_lvl2.reshape(-1,Cls)
        vox_lovasz_label_lvl2 = voxels_lvl2.reshape(-1)
        
        vox_fl_predict_lvl3 = vox_logits_lvl3.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl3 = voxels_lvl3.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl3 = vox_logits_lvl3.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl3 = vox_logits_lvl3.reshape(-1,Cls)
        vox_lovasz_label_lvl3 = voxels_lvl3.reshape(-1)
        
        loss = dict(level0_loss = torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl0,vox_fl_label_lvl0)) + \
                                  torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl0,voxels_lvl0)) + \
                                  torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl0,voxels_lvl0)) + \
                                  torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl0,vox_lovasz_label_lvl0)),
                    level1_loss = 0.5 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl1,vox_fl_label_lvl1)) + \
                                        torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl1,voxels_lvl1)) + \
                                        torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl1,voxels_lvl1)) + \
                                        torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl1,vox_lovasz_label_lvl1))),
                    level2_loss = 0.25 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl2,vox_fl_label_lvl2)) + \
                                          torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl2,voxels_lvl2)) + \
                                          torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl2,voxels_lvl2)) + \
                                          torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl2,vox_lovasz_label_lvl2))),
                    level3_loss = 0.125 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl3,vox_fl_label_lvl3)) + \
                                           torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl3,voxels_lvl3)) + \
                                           torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl3,voxels_lvl3)) + \
                                           torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl3,vox_lovasz_label_lvl3)))
                    )
        return loss
    
    def predict(self, batch_inputs,batch_data_samples):
        """Forward predict function."""
        occ_ori_logits = self._forward(batch_inputs,batch_data_samples)
        B,X,Y,Z,Cls = occ_ori_logits.shape
        if X==256:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
        elif self.occ3d:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_3d_masked'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
        else:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
            
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.eval_ann_info['pts_semantic_mask'] = voxels_lvl0[i].cpu().numpy().astype(np.uint8)

        final_vox_logits = [occ_ori_logit.reshape(-1,Cls) for occ_ori_logit in occ_ori_logits]
        return self.postprocess_result(final_vox_logits, batch_data_samples)
    
    def postprocess_result(self, voxel_logits, batch_data_samples):
        
        for i in range(len(voxel_logits)):
            voxel_logit = voxel_logits[i]
            voxel_pred = voxel_logit.argmax(dim=1)
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': voxel_logit}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': voxel_pred})
            })
        return batch_data_samples
    
    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs, batch_data_samples):
        pass
    

    