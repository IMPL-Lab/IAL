import numpy as np
import os.path as osp
from nuscenes import NuScenes

import mmengine
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
from tools.submit import get_lidar_token, save_test_results
from typing import Optional, Sequence, Union
from nuscenes.nuscenes import NuScenes
DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class SavePredictionHook(Hook):
    """Save prediction masks

    Args:
        save_prediction (bool): If True, save prediction results in folder work_dirs/timestamp/pred_results/
    """    
    def __init__(self,
                 dataset_path: str = 'data/nuscenes_mini',
                 save_instance: bool = False,
                 save_semantic: bool = False,
                 save_panoptic: bool = False,
                 submit_panoptic: bool = False,
                 save_name: str = 'default',
                 taskset: str = 'mini',):
        
        super().__init__()
        self.save_instance = save_instance
        self.save_semantic = save_semantic
        self.save_panoptic = save_panoptic
        self.submit_panoptic = submit_panoptic
        self.save_prediction = save_instance or save_semantic or save_panoptic
        self.learning_map_inv = None
        self.id_offset = None
        self.work_dir = None
        self.save_name = save_name
        if self.save_name == 'lidar_token':
            self.nusc = NuScenes(version='v1.0-'+taskset, dataroot=dataset_path, verbose=True)
    
    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test iteration.

        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='test')
        if self.save_prediction:
            if self.learning_map_inv is None:
                self.learning_map_inv = runner._test_evaluator['learning_map_inv']
            if self.id_offset is None:
                self.id_offset = runner._test_evaluator['id_offset']
            if self.work_dir is None:
                self.work_dir = osp.join(runner.work_dir, runner.timestamp, 'pred_results')

            for (data_sample, output) in zip(data_batch['data_samples'], outputs):
                if runner._test_evaluator.dataset_type == 'semantickitti':
                    lidar_path = data_sample.lidar_path
                    seq = lidar_path.split('/')[-3]
                    frame = lidar_path.split('/')[-1].split('.')[-2]
                elif runner._test_evaluator.dataset_type == 'nuscenes':
                    if hasattr(data_sample, 'lidar_path'):
                        lidar_path = data_sample.lidar_path
                        seq = lidar_path.split('/')[-2]
                        frame = lidar_path.split('/')[-1].split('.')[0]
                        if self.save_name == 'lidar_token':
                            frame = get_lidar_token(self.nusc, data_sample.sample_token)
                    else:
                        lidar_path = 'temp_lidar_path'
                        seq = lidar_path
                        frame = 'temp_frame'
                    
                instance_mask = output.pred_pts_seg.pts_instance_mask
                semantic_mask = output.pred_pts_seg.pts_semantic_mask
                if not isinstance(instance_mask, np.ndarray):
                    instance_mask = instance_mask.cpu().numpy()
                if not isinstance(semantic_mask, np.ndarray):
                    semantic_mask = semantic_mask.cpu().numpy()
                panoptic_mask = self.get_panoptic_mask(semantic_mask, instance_mask)

                if self.save_instance:
                    instance_dir = osp.join(self.work_dir, str(seq).zfill(2), 'instance')
                    mmengine.mkdir_or_exist(instance_dir)
                    np.save(osp.join(instance_dir, f'{frame}.npy'), instance_mask)

                if self.save_semantic:
                    semantic_dir = osp.join(self.work_dir, str(seq).zfill(2), 'semantic')
                    mmengine.mkdir_or_exist(semantic_dir)
                    np.save(osp.join(semantic_dir, f'{frame}.npy'), semantic_mask)
                
                if self.save_panoptic:
                    panoptic_dir = osp.join(self.work_dir, str(seq).zfill(2), 'panoptic')
                    mmengine.mkdir_or_exist(panoptic_dir)
                    np.save(osp.join(panoptic_dir, f'{frame}.npy'), panoptic_mask)

                if self.submit_panoptic:
                    assert runner._test_evaluator.dataset_type == 'nuscenes', 'Only support nuscenes dataset for now'
                    panoptic_dir = osp.join(self.work_dir, str(seq).zfill(2), 'panoptic_submit')
                    mmengine.mkdir_or_exist(panoptic_dir)
                    lidar_token = get_lidar_token(self.nusc, data_sample.sample_token )
                    save_test_results(
                        sem_preds=semantic_mask,
                        ins_preds=instance_mask,
                        output_dir=panoptic_dir,
                        batch=data_batch,
                        class_inv_lut=self.learning_map_inv,
                        tokens=f'{lidar_token}',
                        dataset=runner._test_evaluator.dataset_type
                    )
                    
    def get_panoptic_mask(self, semantic_preds, instance_preds):
        """Compute panoptic mask from semantic and instance masks

        Args:
            semantic_preds (np.array): Shape (# points,)
            instance_preds (np.array): Shape (# points,)

        Returns:
            np.array: Shape (# points,)
        """        
        learning_map_inv_array = np.zeros(len(self.learning_map_inv))
        for i, v in enumerate(self.learning_map_inv.values()):
            learning_map_inv_array[i] = v
        
        semantic_preds_inv = learning_map_inv_array[semantic_preds]
        panoptic_preds = semantic_preds_inv.reshape(-1, 1) + (instance_preds * self.id_offset).reshape(-1, 1)

        return panoptic_preds.reshape(-1).astype(np.uint32)