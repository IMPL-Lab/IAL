
seed = 133365207  
randomness = dict(seed=seed)

custom_hooks = [
    dict(type='SavePredictionHook', 
         taskset='trainval',
         save_name='default',
         dataset_path='/path/to/save/',  
         save_semantic=False, 
         submit_panoptic=False),
]

custom_imports = dict(
    imports=[
        'ial.backbones.lidar_backbone',
        'ial.backbones.swiftnet',
        'ial.data_preprocessors.data_preprocessor_ial',
        'ial.decode_heads.ial_head',
        'ial.segmentors.ial',
        'ial.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'datasets.nuscenes_dataset_seg',
        'datasets.transforms.loading',
        'datasets.transforms.formating',
        'datasets.transforms.transforms_3d',
        'hooks.prediction_hook',
    ],
    allow_failed_imports=False) 