data_root = 'data/nuscenes_full/'

dataset_type = 'NuScenesSegDataset'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='',
    pts_panoptic_mask='',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)

use_intensity = True
vfe_inchannel = 6 if use_intensity else 5
lidar_dim = 4 if use_intensity else 3

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
point_cloud_range_polar = [0, -3.14159265359, -5., 50.0, 3.14159265359, 3.]
all_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
thing_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stuff_class = [11, 12, 13, 14, 15, 16]
ignore_index = 0
num_classes = 17
seg_offset = 1000
grid_shape = [480, 360, 32]

img_scale = (640, 360)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
resnet18_path = 'checkpoints/resnet18-5c106cde.pth'
feature_map_dims = [128]

cand_2d_folder = 'gsam/pred_inst_pts'
img_aug = True

# -------------------------- pipeline -------------------------- #
pre_transform_combo = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=lidar_dim),
    dict(
        type='_LoadViewImage_Mask',
        ),
    dict(type='_ImageResize', img_scale=img_scale, keep_ratio=True),
    dict(type='_ImageyNormalize', **img_norm_cfg),
    dict(
        type='_LoadAnnotations3D',
        cand_2d_folder=cand_2d_folder, 
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.uint8',
        seg_offset=seg_offset,
        dataset_type='nuscenes'),
    dict(type='_PointSegClassMapping')
]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=lidar_dim,
        backend_args=None),
    dict(
        type='_LoadViewImage_Mask',
        ),
    dict(type='_ImageResize', img_scale=img_scale, keep_ratio=True),
    dict(type='_ImageyNormalize', **img_norm_cfg),
    dict(
        type='_LoadAnnotations3D',
        cand_2d_folder=cand_2d_folder, 
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.uint8',
        seg_offset=seg_offset,
        dataset_type='nuscenes',
        backend_args=None),
    dict(type='_PointSegClassMapping', ),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='LaserSwap',
                    img_aug=img_aug,
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-30, 10],
                    pre_transform=[*pre_transform_combo],
                    prob=0.5)
            ],
            [
                dict(
                    type='InstPaste',
                    img_aug=img_aug,
                    instance_classes=thing_class,
                    vertical_swap_ratio=0.5,
                    inst_paste_ratio=1.0,
                    pre_transform=[*pre_transform_combo],
                    prob=0.5)
            ],
        ],
        prob=[0.2, 0.8]),
    dict(
        type='RandomFlip3DMM',
        ),
    dict(
        type='_GlobalRotScaleTransAll',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DMMInputs', 
        keys=['imgs', 'points', 'pts_semantic_mask', 'pts_instance_mask'],
        meta_keys=['scale_factor', 'pcd_scale_factor', 'pcd_trans']
        )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=lidar_dim,
        backend_args=None),
    dict(
        type='_LoadViewImage_Mask',
        ),
    dict(type='_ImageResize', img_scale=img_scale, keep_ratio=True),
    dict(type='_ImageyNormalize', **img_norm_cfg),
    dict(
        type='_LoadAnnotations3D',
        cand_2d_folder=cand_2d_folder,
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.uint8',
        seg_offset=seg_offset,
        dataset_type='nuscenes',
        backend_args=None),
    dict(type='_PointSegClassMapping', ),
    dict(type='Pack3DMMInputs', 
        keys=['imgs', 'points', 'pts_semantic_mask', 'pts_instance_mask'],
        meta_keys=['pcd_scale_factor', 'pcd_trans', 'scale_factor']
        )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True)
)

val_dataloader = test_dataloader 

enable_sync_bn = False
if enable_sync_bn:
    vfe_norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
    c3d_norm_cfg = dict(type='SyncBN', eps=1e-5, momentum=0.1)
else:
    vfe_norm_cfg = dict(type='BN1d', eps=1e-3, momentum=0.01)
    c3d_norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.1)

sample_amount = 256
sample_amount_PQG = 128

inst_3d_params = dict(
    nms_para_3d=dict(
        K=sample_amount_PQG,
        dist_thresh=0.7,
        adapt_thresh=True,
        vote_thresh=0.1,
        score_thresh=0.1,
        pts_range_polar=point_cloud_range_polar,  
        pts_range_cart=point_cloud_range,  
        grid_size=grid_shape  
        ),
    nms_para_fuse=dict(
        K=sample_amount_PQG,
        ),
    hm_sigma=3.,
    loss_heatmap=dict(type='mmdet.MSELoss', loss_weight=0.1),
    loss_offset=dict(type='mmdet.L1Loss', loss_weight=0.1),
    loss_fog=dict(type='mmdet.CrossEntropyLoss', loss_weight=0.01),
    loss_height=dict(type='mmdet.MSELoss', loss_weight=0.01),
)

# -------------------------- model -------------------------- #
model = dict(
    type='IAL',
    dataset_type='nuscenes',
    enable_sync_bn=enable_sync_bn,
    inst_3d_params=inst_3d_params, 
    
    img_backbone=dict(
        type='SwiftNetResNet', 
        layers=[2, 2, 2, 2],
        num_feature=(128, 128, 128),
        build_decoder=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/resnet18-5c106cde.pth')),  
    feat_dims=feature_map_dims,  

    data_preprocessor=dict(
        type='Seg3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        inst_3d_params=inst_3d_params,
        inst_labels=thing_class,  
        grid_shape=grid_shape,  
        return_points=True,
        voxel_layer=dict(
            grid_shape=grid_shape,  
            point_cloud_range=point_cloud_range_polar,  
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        norm_cfg=vfe_norm_cfg,
        in_channels=6,  
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    
    backbone=dict(
        type='Cylinder3D_HM',
        grid_size=grid_shape,  
        input_channels=16,
        base_channels=32,
        more_conv=True,
        out_channels=256,
        norm_cfg=c3d_norm_cfg,
        ),
    
    decode_head=dict(
        type='IALHead',
        inst_3d_params=inst_3d_params,
        num_decoder_layers=6,
        num_queries=256,  
        embed_dims=256,
        cls_channels=(256, 256, len(all_class)),  
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=thing_class,  
        stuff_class=stuff_class,  
        ignore_index=ignore_index,  
        num_classes=len(all_class),  
        point_cloud_range=point_cloud_range_polar,  
        point_cloud_range_cart=point_cloud_range,  
        assigner_zero_layer_cfg=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
        assigner_cfg=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                        dict(type='mmdet.FocalLossCost', gamma=4.0,alpha=0.25,weight=1.0),
                        dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
        sampler_cfg=dict(type='_MaskPseudoSampler'),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_dice=dict(type='mmdet.DiceLoss', loss_weight=2.0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.25,
            loss_weight=1.0),
    ),
    
    train_cfg=None,
    test_cfg=dict(mode='whole'),
) 

# -------------------------- evaluation -------------------------- #
val_evaluator = dict(
    type='_PanopticSegMetric',
    thing_class_inds=thing_class, 
    stuff_class_inds=stuff_class,  
    min_num_points=15,
    id_offset=1000,  
    dataset_type='nuscenes',
    learning_map_inv=dict({ 
        0: 10,
        1: 11,
        2: 15,
        3: 18,
        4: 20,
        5: 30,
        6: 31,
        7: 32,
        8: 40,
        9: 44,
        10: 48,
        11: 49,
        12: 50,
        13: 51,
        14: 70,
        15: 71,
        16: 72,
        17: 80,
        18: 81,
        19: 0
    }))

test_evaluator = val_evaluator
