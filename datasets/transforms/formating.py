# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import InstanceData
from numpy import dtype

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints

from tools.projection import update_lidars2imgs_info

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        data_copy = data.copy() # for debug ValueEeeor: negative stride is not supported
        if data.dtype is dtype('float64'):
            data_copy = data_copy.astype('float32') 
        return torch.from_numpy(data_copy)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class Pack3DMMInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'imgs']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor',
                            'pcd_rotation_angle', 'lidar_path',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global'),
        dataset_type='nuscenes'
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys
        self.dataset_type = dataset_type
        assert dataset_type in ['nuscenes', 'semantickitti'], "dataset_type must be nuscenes or semantickitti"
        self.cam_meta_keys = ('sample_idx', 'token', 'ego2global', 'images', 
                              'lidar_points', 'cam2img', 'lidar2cam', 'ori_cam2img', 
                              'img_shape', 'ori_shape', 'pad_shape',
                              'p_dict', 'tr_mat',
                              )

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'imgs' in results:
            if isinstance(results['imgs'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['imgs'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['imgs'] = imgs # (N_V, C, H, W)
            elif isinstance(results['imgs'], np.ndarray):
                img = results['imgs']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    if len(img.shape) == 3:
                        img = to_tensor(img).permute(2, 0, 1).contiguous()
                    elif len(img.shape) == 4:
                        img = to_tensor(img).permute(0, 3, 1, 2).contiguous()
                else:
                    if len(img.shape) == 3:
                        img = to_tensor(img).permute(2, 0, 1).contiguous()
                    elif len(img.shape) == 4:
                        img = to_tensor(img).permute(0, 3, 1, 2).contiguous()
                results['imgs'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data_sample.set_metainfo(img_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                elif key in ['lidar_token']:
                    pass
                elif key in ['lidar_path']:
                    pass
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
            if 'lidar_token' in self.keys:
                data_sample.eval_ann_info['token'] = results['lidar_token'] # for nuscenes submission
            if 'lidar_path' in self.keys:
                data_sample.eval_ann_info['lidar_path'] = results['lidar_path'] # for semantickitti submission
        else:
            data_sample.eval_ann_info = None

        sample_token = results['token'] 
        data_sample.lidar_path = results['lidar_points']['lidar_path']
        data_sample.sample_token = sample_token  
        
        # load lidar and image meta info
        img_metas = {}
        for key in self.cam_meta_keys:
            if key in results:
                img_metas[key] = results[key]
        img_metas = update_lidars2imgs_info(img_metas, dataset_type=self.dataset_type)
        sample_token = results['token'] 
        data_sample.lidar_path = results['lidar_points']['lidar_path']
        data_sample.sample_token = sample_token
        data_sample.img_metas = img_metas
        aug_result = {}
        if 'lidars2imgs_mix_torch' in results.keys():
            aug_result['lidars2imgs_mix_torch'] = results['lidars2imgs_mix_torch']
        data_sample.aug_result = aug_result
              
        # load 2D query proposals
        if 'sam_pmask_2d' in results.keys():
            cand_2d = {}
            cand_2d['sam_pmask_2d'] = results['sam_pmask_2d']
            cand_2d['sam_pscore_2d'] = results['sam_pscore_2d']
            data_sample.cand_2d = cand_2d
            
        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs
        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

