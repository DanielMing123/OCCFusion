# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Det3DDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class OccFusionDataPreprocessor(Det3DDataPreprocessor):

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']
                
            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict
            
            # Create sparse voxel feature based on the VoxelNet
            batch_sparse_voxel_feats = []
            batch_sparse_voxel_coord = []
            for point, data_sample in zip(inputs['points'], data_samples):
                voxel_coords = data_sample.point_coors
                voxel_coords = voxel_coords[:,[2,0,1]] # put D to first dimension
                voxel_coords, inv_ind, voxel_counts = torch.unique(voxel_coords, dim=0, return_inverse=True, return_counts=True)
                batch_sparse_voxel_coord.append(voxel_coords.long())
                voxel_features = []
                for i in range(len(voxel_coords)):
                    voxel=torch.zeros((35,8))
                    pts = point[inv_ind == i]
                    if voxel_counts[i] > 35:
                        pts = pts[:35, :]
                        voxel_counts[i] = 35
                    
                    # augment the points
                    voxel[:pts.shape[0], :] = torch.cat((pts, pts[:, :3] - torch.mean(pts[:, :3], 0)), dim=1)
                    voxel_features.append(voxel)
                batch_sparse_voxel_feats.append(torch.stack(voxel_features,dim=0).to(inputs['imgs'].device))
            
            batch_inputs['sparse_voxel_feats'] = batch_sparse_voxel_feats
            batch_inputs['sparse_voxel_coords'] = batch_sparse_voxel_coord
        
        
        if 'occ_200' in data['inputs']:
            batch_inputs['dense_occ_200'] = data['inputs']['occ_200']

        if 'occ_3d' in data['inputs']:
            batch_inputs['dense_occ_3d'] = data['inputs']['occ_3d']
            
        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}
    
    @torch.no_grad()
    def voxelize(self, points, data_samples) -> List[Tensor]:
        """Apply voxelization to point cloud. In TPVFormer, it will get voxel-
        wise segmentation label and voxel/point coordinates.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (List[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            List[Tensor]: Coordinates of voxels, shape is Nx3,
        """
        for point, data_sample in zip(points, data_samples):
            min_bound = point.new_tensor(self.voxel_layer.point_cloud_range[:3])
            coors = torch.floor((point[:,:3] - min_bound) / point.new_tensor(self.voxel_layer.voxel_size)).int()
            data_sample.point_coors = coors
