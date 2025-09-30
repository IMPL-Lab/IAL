from typing import List, Optional, Sequence, Any, Dict
import numpy as np
import torch

from mmcv.transforms import BaseTransform, Compose
from mmengine import is_list_of
from mmdet3d.registry import TRANSFORMS

from tools.projection import proj_lidar2img
from tools.img_aug import merge_images_yaw, fit_box_cv, expand_box, crop_box_img, paste_box_img, merge_images_pitch_torch, fit_to_box

@TRANSFORMS.register_module()
class _GlobalRotScaleTransAll(object):
    """Modify: 
    1. rotate along z axis.
    2. save lidar2cam, lidar2img as torch.Tensor.
    """
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        
        input_dict['points'].translate(trans_factor)
        if 'radar' in input_dict:
            input_dict['radar'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor

        if 'bbox3d_fields' in input_dict:
            for key in input_dict['bbox3d_fields']:
                input_dict[key].translate(trans_factor)

        trans_mat = torch.eye(4)
        trans_factor = torch.tensor(trans_factor)
        trans_mat[:3, -1] = trans_factor
        trans_mat_inv = torch.linalg.inv(trans_mat)

        for i in range(len(input_dict["lidar2img"])):
            if isinstance(input_dict["lidar2img"][i], np.ndarray):
                input_dict["lidar2img"][i] = torch.tensor(input_dict["lidar2img"][i])
                input_dict["lidar2cam"][i] = torch.tensor(input_dict["lidar2cam"][i])
            input_dict["lidar2img"][i] = input_dict["lidar2img"][i] @ trans_mat_inv
            input_dict["lidar2cam"][i] = input_dict["lidar2cam"][i] @ trans_mat_inv
        if "lidars2imgs_mix_torch" in input_dict:
            for i in range(len(input_dict["lidars2imgs_mix_torch"])):
                if isinstance(input_dict["lidars2imgs_mix_torch"][i], np.ndarray):
                    input_dict["lidars2imgs_mix_torch"][i] = torch.tensor(input_dict["lidars2imgs_mix_torch"][i], dtype=input_dict["lidar2img"][i].dtype)
                input_dict["lidars2imgs_mix_torch"][i] = input_dict["lidars2imgs_mix_torch"][i] @ trans_mat_inv
        return

    def _rot_bbox_points(self, input_dict, axis):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if 'rot_degree' not in input_dict:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
        else:
            noise_rotation = input_dict['rot_degree']

        assert 'bbox3d_fields' not in input_dict, 'not support rotate bboxes yet'
        if 'rot_degree' not in input_dict:
            rot_mat_T = input_dict['points'].rotate(noise_rotation, axis=axis)
            if 'radar' in input_dict:
                input_dict['radar'].rotate(noise_rotation, axis=axis)
        else:
            rot_mat_T = input_dict['points'].rotate(-noise_rotation, axis=axis)
            if 'radar' in input_dict:
                input_dict['radar'].rotate(-noise_rotation, axis=axis)

        rot_mat = torch.eye(4)
        rot_mat[:3, :3].copy_(rot_mat_T.T)
        rot_mat_inv = torch.inverse(rot_mat)

        for i in range(len(input_dict["lidar2img"])):
            if isinstance(input_dict["lidar2img"][i], np.ndarray):
                input_dict["lidar2img"][i] = torch.tensor(input_dict["lidar2img"][i])
                input_dict["lidar2cam"][i] = torch.tensor(input_dict["lidar2cam"][i])
            input_dict["lidar2img"][i] = input_dict["lidar2img"][i].float() @ rot_mat_inv
            input_dict["lidar2cam"][i] = input_dict["lidar2cam"][i].float() @ rot_mat_inv
        if "lidars2imgs_mix_torch" in input_dict:
            for i in range(len(input_dict["lidars2imgs_mix_torch"])):
                if isinstance(input_dict["lidars2imgs_mix_torch"][i], np.ndarray):
                    input_dict["lidars2imgs_mix_torch"][i] = torch.tensor(input_dict["lidars2imgs_mix_torch"][i], dtype=input_dict["lidar2img"][i].dtype)
                input_dict["lidars2imgs_mix_torch"][i] = input_dict["lidars2imgs_mix_torch"][i] @ rot_mat_inv

        return

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points
        
        if 'radar' in input_dict:
            input_dict['radar'].scale(scale)
            
        if 'bbox3d_fields' in input_dict:
            for key in input_dict['bbox3d_fields']:
                input_dict[key].scale(scale)

        scale_mat = torch.tensor(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32
        )
        scale_mat_inv = torch.inverse(scale_mat)

        for i in range(len(input_dict["lidar2img"])):
            if isinstance(input_dict["lidar2img"][i], np.ndarray):
                input_dict["lidar2img"][i] = torch.tensor(input_dict["lidar2img"][i])
                input_dict["lidar2cam"][i] = torch.tensor(input_dict["lidar2cam"][i])
            input_dict["lidar2img"][i] = input_dict["lidar2img"][i].float() @ scale_mat_inv
            input_dict["lidar2cam"][i] = input_dict["lidar2cam"][i].float() @ scale_mat_inv
        if "lidars2imgs_mix_torch" in input_dict:
            for i in range(len(input_dict["lidars2imgs_mix_torch"])):
                if isinstance(input_dict["lidars2imgs_mix_torch"][i], np.ndarray):
                    input_dict["lidars2imgs_mix_torch"][i] = torch.tensor(input_dict["lidars2imgs_mix_torch"][i], dtype=input_dict["lidar2img"][i].dtype)
                input_dict["lidars2imgs_mix_torch"][i] = input_dict["lidars2imgs_mix_torch"][i] @ scale_mat_inv
        return
    
    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict, rot_axis=2):
        """Function to rotate, scale and translate points.
        """

        self._rot_bbox_points(input_dict, axis=rot_axis)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str

@TRANSFORMS.register_module()
class RandomFlip3DMM:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = torch.eye(3)
        if flip_horizontal:
            rotation = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()
        
        rotation_pad = torch.eye(4)
        rotation_pad[:3, :3] = rotation
        rotation_inv_pad = torch.inverse(rotation_pad)
        for i in range(len(data["lidar2img"])):
            if isinstance(data["lidar2img"][i], np.ndarray):
                data["lidar2img"][i] = torch.tensor(data["lidar2img"][i])
                data["lidar2cam"][i] = torch.tensor(data["lidar2cam"][i])
            data["lidar2img"][i] = data["lidar2img"][i].float() @ rotation_inv_pad
            data["lidar2cam"][i] = data["lidar2cam"][i].float() @ rotation_inv_pad
        if "lidars2imgs_mix_torch" in data:
            for i in range(len(data["lidars2imgs_mix_torch"])):
                if isinstance(data["lidars2imgs_mix_torch"][i], np.ndarray):
                    data["lidars2imgs_mix_torch"][i] = torch.tensor(data["lidars2imgs_mix_torch"][i], dtype=data["lidar2img"][i].dtype)
                data["lidars2imgs_mix_torch"][i] = data["lidars2imgs_mix_torch"][i] @ rotation_inv_pad
        return data

@TRANSFORMS.register_module(force=True)
class InstPaste(BaseTransform):
    def __init__(self,
                 img_aug: bool,
                 instance_classes: List[int],
                 vertical_swap_ratio: float = 0.5,
                 inst_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.vertical_swap_ratio = vertical_swap_ratio
        self.inst_paste_ratio = inst_paste_ratio
        self.prob = prob
        self.pre_transform = Compose(pre_transform)

    def inst_mix_transform(self, input_dict: dict, mix_results: dict, angle=None) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        points = input_dict['points']
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        block_cp = []
        block_cp_instlabel, block_cp_semlabel = [], []
        out_im_list = []
        binary_mask = torch.ones(len(points))
        binary_mask_orig = torch.ones(len(points))
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            pts_instance_mask = input_dict['pts_instance_mask']
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16)
            mix_panoptic = True
            
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            sam_pmask, sam_pscore = input_dict['sam_pmask_2d'], input_dict['sam_pscore_2d']
            sam_pmask_mix, sam_pscore_mix = mix_results['sam_pmask_2d'], mix_results['sam_pscore_2d']
            sam_pmask_mix[sam_pmask_mix>0] += (1000<<16)
            mix_sam_pred = True
            
        if np.random.random() < self.vertical_swap_ratio:
            """Follow PolarMix, we apply scene-level vertical swap along with instance-level paste."""

            start_angle = (np.random.random() - 1) * np.pi
            end_angle = start_angle + np.pi
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:,
                                                                            0])

            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            a, b = points[idx].shape[0], mix_points[mix_idx].shape[0]
            block_orig = points[idx].coord[:, :3]
            
            swap_points = points.cat([points[idx], mix_points[mix_idx]])
            swap_pts_semantic_mask = np.concatenate(
                (pts_semantic_mask[idx.numpy()],
                 mix_pts_semantic_mask[mix_idx.numpy()]),
                axis=0)
            
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                    (pts_instance_mask[idx.numpy()],
                    mix_instance_mask[mix_idx.numpy()]),
                    axis=0)  
                
            if mix_sam_pred:  
                sam_pmask = np.concatenate(
                    (sam_pmask[idx.numpy()],
                    sam_pmask_mix[mix_idx.numpy()]),
                    axis=0)
                sam_pscore = np.concatenate(
                    (sam_pscore[idx.numpy()],
                    sam_pscore_mix[mix_idx.numpy()]),
                    axis=0)    
            
            points = swap_points
            pts_semantic_mask = swap_pts_semantic_mask
            binary_mask = torch.ones(len(points))
            binary_mask[a:a+b] = 0
            binary_mask_orig[a:] = 0

            if self.img_aug:
                image_list_orig = input_dict['imgs']
                image_list_mix = mix_results['imgs']
                image_list_orig = [torch.tensor(img) for img in image_list_orig]
                image_list_mix  = [torch.tensor(img) for img in image_list_mix]
                
                lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
                
                img_size = input_dict['img_shape'][:2]
                ori_size = input_dict['ori_shape']
                img_scale = input_dict['scale_factor'][0]
                assert int(img_size[0]) == int(ori_size[0] * img_scale)
                
                for v in range(len(image_list_orig)):
                    im_orig, im_mix = image_list_orig[v].clone(), image_list_mix[v].clone()
                    block_points_img, mask = proj_lidar2img(block_orig, lidar2img_orig[v], 
                                                            img_size=(ori_size[1], ori_size[0]), 
                                                            min_dist=1.0)
                    block_points_img = block_points_img * img_scale
                    l, r = fit_to_box(block_points_img.numpy(), 'vertical')

                    img_new = merge_images_yaw(im_orig, im_mix, l, r)
                    out_im_list.append(img_new)
                input_dict['imgs'] = out_im_list
            
        if np.random.random() < self.inst_paste_ratio:
            instance_points, instance_pts_semantic_mask = [], []
            if mix_panoptic:
                instance_pts_instance_mask = []
            if mix_sam_pred:
                instance_pts_sam_pmask, instance_pts_sam_pscore = [], []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
                if mix_panoptic:
                    instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
                if mix_sam_pred:
                    instance_pts_sam_pmask.append(sam_pmask_mix[mix_idx])
                    instance_pts_sam_pscore.append(sam_pscore_mix[mix_idx])
            
            block_cp = instance_points.copy()
            block_cp_instlabel = instance_pts_instance_mask.copy()
            block_cp_semlabel  = instance_pts_semantic_mask.copy()
            
            instance_points = mix_points.cat(instance_points)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)
            if mix_panoptic:
               instance_pts_instance_mask = np.concatenate(
                instance_pts_instance_mask, axis=0)
            if mix_sam_pred:
                instance_pts_sam_pmask = np.concatenate(
                    instance_pts_sam_pmask, axis=0)
                instance_pts_sam_pscore = np.concatenate(
                    instance_pts_sam_pscore, axis=0) 

            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            if mix_panoptic:
                copy_pts_instance_mask = [instance_pts_instance_mask]
            if mix_sam_pred:
                copy_pts_sam_pmask = [instance_pts_sam_pmask]
                copy_pts_sam_pscore = [instance_pts_sam_pscore]
                
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)
            if mix_panoptic:
                copy_pts_instance_mask = np.concatenate(
                copy_pts_instance_mask, axis=0)
            if mix_sam_pred:
                copy_pts_sam_pmask = np.concatenate(
                    copy_pts_sam_pmask, axis=0)
                copy_pts_sam_pscore = np.concatenate(
                    copy_pts_sam_pscore, axis=0)

            c = copy_points.shape[0]
            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                (pts_instance_mask, copy_pts_instance_mask), axis=0)
            if mix_sam_pred:
                sam_pmask = np.concatenate(
                    (sam_pmask, copy_pts_sam_pmask), axis=0)
                sam_pscore = np.concatenate(
                    (sam_pscore, copy_pts_sam_pscore), axis=0)
            
            binary_mask = torch.cat([binary_mask, torch.zeros(c)])
            binary_mask_orig = binary_mask_orig

            if self.img_aug:
                if len(out_im_list) == 0:
                    out_im_list = input_dict['imgs']
                
                image_list_orig = out_im_list
                image_list_mix = mix_results['imgs']
                image_list_orig = [torch.tensor(img) for img in image_list_orig]
                image_list_mix  = [torch.tensor(img) for img in image_list_mix]
                
                img_size = input_dict['img_shape'][:2]
                ori_size = input_dict['ori_shape']
                img_scale = input_dict['scale_factor'][0]
                assert int(img_size[0]) == int(ori_size[0] * img_scale)
                
                lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
                
                cp_coord = torch.cat([inst.coord for inst in block_cp])
                cp_instlabel = np.concatenate(block_cp_instlabel)
                cp_semlabel  = np.concatenate(block_cp_semlabel)
                N = binary_mask.shape[0]
                N_cp = cp_instlabel.shape[0]
                N_ro = 0
                
                if N_cp > 0:
                    
                    for i in range(len(image_list_orig)):
                        im_orig = image_list_orig[i].numpy()
                        im_1_copy = im_orig.copy()
                        im_mix = image_list_mix[i].numpy()
                        cp_coord_img, mask = proj_lidar2img(cp_coord, lidar2img_mix[i], 
                                                            img_size=(ori_size[1], ori_size[0]), 
                                                            min_dist=1.0)
                        cp_coord_img = cp_coord_img * img_scale

                        cp_coord_img = cp_coord_img.numpy()
                        
                        cp_instlabel_v = cp_instlabel[mask.detach().cpu().numpy()]
                        cp_semlabel_v  = cp_semlabel[mask.detach().cpu().numpy()]
                        
                        for inst_id in np.unique(cp_instlabel_v):
                            inst_coord = cp_coord_img[cp_instlabel_v == inst_id]
                            inst_cls = cp_semlabel_v[cp_instlabel_v == inst_id][0]
                            box = fit_box_cv(inst_coord)
                            box_ex = expand_box(box, 10, im_mix.shape[:2])
                            crop_img = crop_box_img(box_ex, im_mix)
                            im_1_copy = paste_box_img(box_ex, im_orig, crop_img)

                        out_im_list[i] = im_1_copy
        
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        if mix_sam_pred:
            input_dict['sam_pmask_2d'] = sam_pmask
            input_dict['sam_pscore_2d'] = sam_pscore

        if self.img_aug:
            input_dict['imgs'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']
        index = np.random.randint(0, len(dataset))
        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            mix_results.update({'dataset': dataset})
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.inst_mix_transform(input_dict, mix_results)
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'vertical_swap_ratio={self.vertical_swap_ratio}, '
        repr_str += f'inst_paste_ratio={self.inst_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module(force=True)
class LaserSwap(BaseTransform):

    def __init__(self,
                 img_aug: bool,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        assert is_list_of(num_areas, int), \
            'num_areas should be a list of int.'
        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def scene_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        binary_mask = None
        binary_mask_orig = torch.zeros(len(points), dtype=torch.uint8)


        rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
        pitch = torch.atan2(points.coord[:, 2], rho)
        pitch = torch.clamp(pitch, self.pitch_angles[0] + 1e-5,
                            self.pitch_angles[1] - 1e-5)

        mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 +
                             mix_points.coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, self.pitch_angles[0] + 1e-5,
                                self.pitch_angles[1] - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(self.pitch_angles[1], self.pitch_angles[0],
                                 num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []
        block_orig, block_mix = [], []
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            pts_instance_mask = input_dict['pts_instance_mask']
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16)
            out_pts_instance_mask = []
            mix_panoptic = True
        
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            sam_pmask, sam_pscore = input_dict['sam_pmask_2d'], input_dict['sam_pscore_2d']
            sam_pmask_mix, sam_pscore_mix = mix_results['sam_pmask_2d'], mix_results['sam_pscore_2d']
            sam_pmask_mix[sam_pmask_mix>0] += (1000<<16)
            out_sam_pmask, out_sam_pscore = [], []
            mix_sam_pred = True
            
        for i in range(num_areas):
            start_angle = angle_list[i + 1] / 180 * np.pi
            end_angle = angle_list[i] / 180 * np.pi
            if i % 2 == 0:
                idx = (pitch > start_angle) & (pitch <= end_angle)
                selected_points = points[idx]
                
                out_points.append(points[idx])
                block_orig.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(pts_instance_mask[idx.numpy()])
                if mix_sam_pred:
                    out_sam_pmask.append(sam_pmask[idx.numpy()])
                    out_sam_pscore.append(sam_pscore[idx.numpy()])
                if binary_mask == None:
                    binary_mask = torch.ones(selected_points.shape[0], dtype=torch.uint8)
                else:
                    binary_mask = torch.cat([binary_mask, torch.ones(selected_points.shape[0], dtype=torch.uint8)])
                binary_mask_orig[idx] = 1  
                
            else:
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                selected_points = mix_points[idx]
                
                out_points.append(mix_points[idx])
                block_mix.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(mix_instance_mask[idx.numpy()])
                if mix_sam_pred:
                    out_sam_pmask.append(sam_pmask_mix[idx.numpy()])
                    out_sam_pscore.append(sam_pscore_mix[idx.numpy()])
                binary_mask = torch.cat([binary_mask, torch.zeros(selected_points.shape[0], dtype=torch.uint8)])

        
        if self.img_aug:
            out_im_list = []
            image_list_orig = input_dict['imgs']
            image_list_mix = mix_results['imgs']
            image_list_orig = [torch.tensor(img) for img in image_list_orig] if isinstance(image_list_orig[0], np.ndarray) else image_list_orig
            image_list_mix  = [torch.tensor(img) for img in image_list_mix] if isinstance(image_list_mix[0], np.ndarray) else image_list_mix
            
            img_size = input_dict['img_shape'][:2]
            ori_size = input_dict['ori_shape']
            img_scale = input_dict['scale_factor'][0]
            assert int(img_size[0]) == int(ori_size[0] * img_scale)
            
            lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
            
            for v in range(len(image_list_orig)):
                im_orig, im_mix = image_list_orig[v].clone(), image_list_mix[v].clone()
                for i in range(len(block_mix)):
                    pblock = block_mix[i].coord
                    points_img_b, mask_b = proj_lidar2img(pblock, lidar2img_mix[v], 
                                                        img_size=(ori_size[1], ori_size[0]), 
                                                        min_dist=1.0)
                    points_img_b = points_img_b * img_scale
                    mixed_img, block = merge_images_pitch_torch(im_orig, im_mix, points_img_b)   
                    im_orig = mixed_img
                out_im_list.append(im_orig)
            
                
            
        out_points = points.cat(out_points)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        
        assert binary_mask.shape[0] == out_points.coord.shape[0]
        assert torch.count_nonzero(binary_mask) == torch.count_nonzero(binary_mask_orig)
        
        if mix_panoptic:
            out_pts_instance_mask = np.concatenate(out_pts_instance_mask, axis=0)
            input_dict['pts_instance_mask'] = out_pts_instance_mask
        if mix_sam_pred:
            out_sam_pmask = np.concatenate(out_sam_pmask, axis=0)
            out_sam_pscore = np.concatenate(out_sam_pscore, axis=0)
            input_dict['sam_pmask_2d'] = out_sam_pmask
            input_dict['sam_pscore_2d'] = out_sam_pscore
        if self.img_aug:
            input_dict['imgs'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through LaserMix, while not found.'
        dataset = input_dict['dataset']

        index = np.random.randint(0, len(dataset))
        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            mix_results.update({'dataset': dataset})
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')
        
        input_dict = self.scene_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'pitch_angles={self.pitch_angles}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str
