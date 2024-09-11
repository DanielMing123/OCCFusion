_base_ = ['_base_/default_runtime.py']
custom_imports = dict(imports=['occfusion'], allow_failed_imports=False)

load_from = 'ckpt/OccFusion_Cam_Lidar_semanticKITTI_ckpt_new_eval/epoch_15.pth'

dataset_type = 'SemanticKittiSegDataset'
data_root = 'data/semantickitti'

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
grid_size_vt = [128, 128, 16]
num_points_per_voxel = 35
nbr_class = 20 # 20
use_lidar=True
use_radar=False 
use_occ3d=False
find_unused_parameters=False

model = dict(
    type='OccFusion',
    use_occ3d=use_occ3d,
    use_lidar=use_lidar,
    use_radar=use_radar,
    data_preprocessor=dict(
        type='OccFusionDataPreprocessor',
        pad_size_divisor=32,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        voxel=True,
        voxel_layer=dict(
            grid_shape=grid_size_vt, 
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        )
        ), 
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),  
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    view_transformer=dict(
        type='MultiScaleInverseMatrixVT',
        feature_strides=[8, 16, 32],
        in_channel=[32,64,128,256],
        grid_size=[[128, 128, 16],
                   [64, 64, 8],
                   [32, 32, 4]],
        x_bound=[0, 51.2],
        y_bound=[-25.6, 25.6],
        z_bound=[-2., 4.4],
        sampling_rate=[4,5,6],
        num_cams=[None,None,None],
        enable_fix=False,
        use_lidar=use_lidar,
        use_radar=use_radar
        ),
    svfe_lidar=dict(
        type='SVFE',
        num_pts=num_points_per_voxel,
        input_dim=7,
        grid_size=grid_size_vt
        ),
    svfe_radar=dict(
        type='SVFE',
        num_pts=num_points_per_voxel,
        input_dim=11,
        grid_size=grid_size_vt
        ),
    occ_head=dict(
        type='OccHead',
        channels=[32,64,128,256],
        num_classes=nbr_class
        )
)

train_pipeline = [
    dict(
        type='SemanticKITTI_Image_Load',
        to_float32=False,
        color_type='unchanged',
        backend_args=backend_args),
    dict(type='LoadSemanticKITTI_Occupancy'),
    dict(type='LoadSemanticKITTI_Lidar',
         pc_range=point_cloud_range),
    dict(
        type='MultiViewWrapper',
        transforms=dict(type='PhotoMetricDistortion3D')),
    dict(
        type='Custom3DPack',
        keys=['img','occ_semantickitti_masked','points'],
        meta_keys=['lidar2img'])
]

val_pipeline = [
    dict(
        type='SemanticKITTI_Image_Load',
        to_float32=False,
        color_type='unchanged',
        backend_args=backend_args),
    dict(type='LoadSemanticKITTI_Occupancy'),
    dict(type='LoadSemanticKITTI_Lidar',
         pc_range=point_cloud_range),
    dict(
        type='Custom3DPack',
        keys=['img','occ_semantickitti_masked','points'],
        meta_keys=['lidar2img'])
]

test_pipeline = val_pipeline



train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # data_prefix=data_prefix,
        ann_file='semantickitti_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # data_prefix=data_prefix,
        ann_file='semantickitti_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True)) # True

test_dataloader = val_dataloader


val_evaluator = dict(type='EvalMetric')

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01), # weight_decay=0.01, lr = 5e-5
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=5000, norm_type=2), # try 5000 next time, default=35
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24,val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))