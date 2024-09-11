# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class SemanticKittiSegDataset(BaseDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    METAINFO = {
        'classes':
        ('IoU', 'car', 'bicycle','motorcycle','truck','other-vehicle','person','bicyclist','motorcyclist','road',
         'parking','sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign'),
        'ignore_index':
            255,
        'label_mapping':
        dict([(0,0),(1,0),(10,1),(11,2),(13,5),(15,3),(16,5),(18,4),(20,5),(30,6),(31,7),
              (32,8),(40,9),(44,10),(48,11),(49,12),(50,13),(51,14),(52,0),(60,9),(70,15),
              (71,16),(72,17),(80,18),(81,19),(99,0),(252,1),(253,7),(254,6),(255,8),(256,5),
              (257,5),(258,4),(259,5)])
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        metainfo = dict(label2cat={
            i: cat_name
            for i, cat_name in enumerate(self.METAINFO['classes'])
        })
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        data_list = []
        info['img_path'] = osp.join(self.data_prefix['img_path'],info['img_path'])
        info['voxel_gt_path'] = osp.join(self.data_prefix['img_path'],info['voxel_gt_path'])
        info['voxel_invalid_path'] = osp.join(self.data_prefix['img_path'],info['voxel_invalid_path'])
        info['calib_path'] = osp.join(self.data_prefix['img_path'],info['calib_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['label_mapping'] = self.metainfo['label_mapping']

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode:
            info['eval_ann_info'] = dict()

        data_list.append(info)
        return data_list
