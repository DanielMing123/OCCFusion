# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Union
import cv2
import mmcv
import numpy as np
import torch
import os
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles, Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS
from nuscenes.utils.data_classes import RadarPointCloud


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, lidar2img = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam'],dtype=np.float64)
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'],dtype=np.float64)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # imgs = [
        #     cv2.resize(mmcv.imfrombytes(img_byte, flag=self.color_type,backend='cv2'),(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
        #     for img_byte in img_bytes
        # ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 0.2 # 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class SegLabelMapping(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = np.vectorize(
            label_mapping.__getitem__, otypes=[np.uint8])(
                pts_semantic_mask)

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadOccupancy(BaseTransform):
    
    def transform(self, results: dict) -> dict:
        occ_file_name = results['lidar_points']['lidar_path'].split('/')[-1] + '.npy'
        occ_200_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'occ_samples'
        # occ_3d_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'Occ3D'
        occ_200_path = os.path.join(occ_200_folder, occ_file_name)
        # occ_3d_path = os.path.join(occ_3d_folder, results['token'], 'labels.npz')
        occ_200 = np.load(occ_200_path)
        # occ_3d = np.load(occ_3d_path)
        # occ_3d_semantic = occ_3d['semantics']
        # occ_3d_cam_mask = occ_3d['mask_camera']
        # occ_3d_semantic[occ_3d_semantic==0]=18
        # occ_3d_gt = occ_3d_semantic * occ_3d_cam_mask
        # occ_3d_gt[occ_3d_gt==0]=255
        # occ_3d_gt[occ_3d_gt==17]=0
        # occ_3d_gt[occ_3d_gt==18]=17
        # occ_3d_gt = torch.from_numpy(occ_3d_gt)
        # idx = torch.where(occ_3d_gt > 0)
        # label = occ_3d_gt[idx[0],idx[1],idx[2]]
        # occ_3d = torch.stack([idx[0],idx[1],idx[2],label],dim=1).float()
        # occ_3d = occ_3d.long()
        
        occ_200[:,3][occ_200[:,3]==0]=255
        occ_200 = torch.from_numpy(occ_200)
        results['occ_200'] = occ_200
        # results['occ_3d'] = occ_3d
        
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadRadarPointsMultiSweeps(BaseTransform):
   """Load radar points from multiple sweeps.
   This is usually used for nuScenes dataset to utilize previous sweeps.
   Args:
       sweeps_num (int): Number of sweeps. Defaults to 10.
       load_dim (int): Dimension number of the loaded points. Defaults to 5.
       use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
   """


   def __init__(self,
                load_dim=18,
                use_dim=[0, 1, 2, 8, 9, 18],
                sweeps_num=5,
                pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]):
       self.load_dim = load_dim
       self.use_dim = use_dim
       self.sweeps_num = sweeps_num
       self.pc_range = pc_range


   def _load_points(self, pts_filename):
       """Private function to load point clouds data.
       Args:
           pts_filename (str): Filename of point clouds data.
       Returns:
           np.ndarray: An array containing point clouds data.
           [N, 18]
       """
       radar_obj = RadarPointCloud.from_file(pts_filename)


       #[18, N]
       points = radar_obj.points


       return points.transpose().astype(np.float32)


   def __call__(self, results):
       """Call function to load multi-sweep point clouds from files.
       Args:
           results (dict): Result dict containing multi-sweep point cloud \
               filenames.
       Returns:
           dict: The result dict containing the multi-sweep points data. \
               Added key and value are described below.
               - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                   cloud arrays.
       """
       radars_dict = results['radars']


       points_sweep_list = []
       for key, sweeps in radars_dict.items():
           if len(sweeps) < self.sweeps_num:
               idxes = list(range(len(sweeps)))
           else:
               idxes = list(range(self.sweeps_num))
          
           ts = sweeps[0]['timestamp'] * 1e-6
           for idx in idxes:
               sweep = sweeps[idx]


               points_sweep = self._load_points(sweep['data_path'])
               points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)


               timestamp = sweep['timestamp'] * 1e-6
               time_diff = ts - timestamp
               time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff


               # velocity compensated by the ego motion in sensor frame
               velo_comp = points_sweep[:, 8:10]
               velo_comp = np.concatenate(
                   (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
               velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
               velo_comp = velo_comp[:, :2]


               # velocity in sensor frame
               velo = points_sweep[:, 6:8]
               velo = np.concatenate(
                   (velo, np.zeros((velo.shape[0], 1))), 1)
               velo = velo @ sweep['sensor2lidar_rotation'].T
               velo = velo[:, :2]


               points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                   'sensor2lidar_rotation'].T
               points_sweep[:, :3] += sweep['sensor2lidar_translation']


               points_sweep_ = np.concatenate(
                   [points_sweep[:, :6], velo,
                    velo_comp, points_sweep[:, 10:],
                    time_diff], axis=1)
               points_sweep_list.append(points_sweep_)
      
       points = np.concatenate(points_sweep_list, axis=0)
      
       points = points[:, self.use_dim]
      
       points = torch.from_numpy(points)
      
       results['radars'] = points
       return self.transform(results)


   def transform(self, results):
       radar_pts = results['radars']
       radar_pts_xyz = radar_pts[:,0:3]
       idx = torch.where((radar_pts_xyz[:,0] > self.pc_range[0])
                         & (radar_pts_xyz[:,1] > self.pc_range[1])
                         & (radar_pts_xyz[:,2] > self.pc_range[2])
                         & (radar_pts_xyz[:,0] < self.pc_range[3])
                         & (radar_pts_xyz[:,1] < self.pc_range[4])
                         & (radar_pts_xyz[:,2] < self.pc_range[5]))
       radar_pts = radar_pts[idx]
       results['radars'] = radar_pts
       return results
  
   def __repr__(self):
       """str: Return a string that describes the module."""
       return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
