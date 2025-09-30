# Copyright (c) OpenMMLab. All rights reserved.
# Modified from mmdetection3d.
from typing import Dict, List, Optional
from mmengine.logging import MMLogger
import mmengine
from ..functional.panoptic_seg_eval import panoptic_seg_eval
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import SegMetric
import os
import json
import numpy as np
from typing import Dict, Optional, Sequence
from nuscenes import NuScenes
from tools.submit import get_lidar_token

@METRICS.register_module()
class _PanopticSegMetric(SegMetric):
    def __init__(self,
                 thing_class_inds: List[int],
                 stuff_class_inds: List[int],
                 min_num_points: int,
                 id_offset: int,
                 dataset_type: str,
                 dataset_path: str = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 taskset: str = None,
                 learning_map_inv = None, 
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.dataset_type=dataset_type
        self.taskset = taskset
        self.learning_map_inv = learning_map_inv
        if submission_prefix:
            assert (taskset is not None) and (dataset_path is not None)
        if taskset == 'test':
            self.nusc = NuScenes(version='v1.0-'+taskset, dataroot=dataset_path, verbose=True)
        
        super(_PanopticSegMetric, self).__init__(
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            prefix=prefix,
            collect_device=collect_device,
            **kwargs)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix: # will report a error when finishing.
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']
        classes = self.dataset_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        include = self.thing_class_inds + self.stuff_class_inds

        gt_labels = []
        seg_preds = []
        
        for i, (eval_ann, sinlge_pred_results) in enumerate(results):
            gt_labels.append(eval_ann) # includes: pts_semantic_mask, pts_instance_mask
            seg_preds.append(sinlge_pred_results)

        ret_dict = panoptic_seg_eval(gt_labels, seg_preds, classes,
                                     thing_classes, stuff_classes, include, self.dataset_type,
                                     self.min_num_points, self.id_offset,
                                     label2cat, ignore_index, logger)
        return ret_dict

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        mmengine.mkdir_or_exist(submission_prefix)

        if self.dataset_type == 'nuscenes':
            meta_dir = os.path.join(submission_prefix, self.taskset)
            mmengine.mkdir_or_exist(meta_dir)
            meta =  {"meta": {
                "task": "segmentation",
                "use_camera": True, 
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False}}
            output = open(os.path.join(submission_prefix, self.taskset, 'submission.json'), 'w')
            json_meta = json.dumps(meta)
            output.write(json_meta)
            output.close()

            for i, (eval_ann, result) in enumerate(results):
                sample_token = eval_ann['token']
                lidar_token = get_lidar_token(self.nusc, sample_token)
                results_dir = os.path.join(submission_prefix, 'panoptic', self.taskset) # file name: <lidar_token>_panoptic.npz
                mmengine.mkdir_or_exist(results_dir)
                results_dir = os.path.join(results_dir, lidar_token) 
                pred_semantic_mask = result['pts_semantic_mask']
                pred_instance_mask = result['pts_instance_mask']
                pred_panoptic_mask = (pred_instance_mask + pred_semantic_mask*self.id_offset).astype(np.uint16)
                curr_file = results_dir +  "_panoptic.npz"
                np.savez_compressed(curr_file, data=pred_panoptic_mask)
        elif self.dataset_type == "semantickitti":
            results_dir = os.path.join(submission_prefix, 'sequences')
            mmengine.mkdir_or_exist(results_dir)
            for i in range(0,22):
                sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
                mmengine.mkdir_or_exist(sub_dir)

            learning_map_inv_array = np.zeros(len(self.learning_map_inv))
            for i, v in enumerate(self.learning_map_inv.values()):
                learning_map_inv_array[i] = v
            for i, (eval_ann, result) in enumerate(results):
                semantic_preds = result['pts_semantic_mask']
                instance_preds = result['pts_instance_mask']
                semantic_preds_inv = learning_map_inv_array[semantic_preds]
                panoptic_preds = semantic_preds_inv.reshape(-1, 1) + (instance_preds * self.id_offset).reshape(-1, 1)

                lidar_path = eval_ann['lidar_path']
                seq = lidar_path.split('/')[-3]
                pts_fname = lidar_path.split('/')[-1].split('.')[-2]+'.label'
                fname = os.path.join(results_dir, seq, 'predictions', pts_fname)
                panoptic_preds.reshape(-1).astype(np.uint32).tofile(fname)
        else:
            raise NotImplementedError()
       
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Replace  pts_semantic/instance_mask with its point-sampled results.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            
            if 'gt_pts_seg' in data_sample and 'pts_instance_mask' in data_sample['gt_pts_seg']: # for evaluation
                eval_info = data_sample['eval_ann_info']
                eval_info['pts_instance_mask'] = np.array(data_sample['gt_pts_seg']['pts_instance_mask'].cpu())
                eval_info['pts_semantic_mask'] = np.array(data_sample['gt_pts_seg']['pts_semantic_mask'].cpu())
            else:
                eval_info = dict() # for submission, no gt provided.
            eval_info['token'] = data_sample['sample_token']
            
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_info, cpu_pred_3d))
 
  