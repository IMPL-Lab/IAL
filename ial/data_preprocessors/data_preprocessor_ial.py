
import numpy as np
import torch
from torch.nn import functional as F
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
import torch_scatter

from .data_preprocessor import _Det3DDataPreprocessor
from tools.query_generator import PanopticLabelGenerator, nb_process_inst
from tools.general_tool import remap_instance_mask
   

@MODELS.register_module(force=True)
class Seg3DDataPreprocessor(_Det3DDataPreprocessor):
    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'cylindrical',
                 inst_3d_params: Optional[dict] = None,
                 inst_labels: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 ignore_label: int = 0,
                 grid_shape: List[int] = [480, 320, 32],
                 return_points: bool = False,
                 voxel_layer: OptConfigType = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(Seg3DDataPreprocessor, self).__init__(
            voxel=voxel,
            voxel_type=voxel_type,
            voxel_layer=voxel_layer,
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            batch_augments=batch_augments)
        
        self.return_points = return_points
        self.inst_labels = inst_labels
        self.ignore_label = ignore_label
        self.hm_sigma = inst_3d_params['hm_sigma']
        self.grid_shape = grid_shape
    
    @torch.no_grad()
    def voxelize(self, points: List[torch.Tensor],
                 data_samples: SampleList) -> Dict[str, torch.Tensor]:
        """
            Compared to _Det3DDataPreprocessor.voxelize,
            this method also generates heatmaps and fog masks for query generation.
        """

        voxel_dict = dict()
        assert self.voxel_type == 'cylindrical', 'Only support cylindrical voxelization for heatmap generation'
        
        voxels, coors = [], []
        heatmaps, fogs = [], []
        
        for i, (res, data_sample) in enumerate(zip(points, data_samples)):
            rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
            phi = torch.atan2(res[:, 1], res[:, 0])
            polar_res = torch.stack((rho, phi, res[:, 2]), dim=-1)
            min_bound = polar_res.new_tensor(
                self.voxel_layer.point_cloud_range[:3])
            max_bound = polar_res.new_tensor(
                self.voxel_layer.point_cloud_range[3:])
            try:  # only support PyTorch >= 1.9.0
                polar_res_clamp = torch.clamp(polar_res, min_bound,
                                                max_bound)
            except TypeError:
                polar_res_clamp = polar_res.clone()
                for coor_idx in range(3):
                    polar_res_clamp[:, coor_idx][
                        polar_res[:, coor_idx] >
                        max_bound[coor_idx]] = max_bound[coor_idx]
                    polar_res_clamp[:, coor_idx][
                        polar_res[:, coor_idx] <
                        min_bound[coor_idx]] = min_bound[coor_idx]
            res_coors = torch.floor(
                (polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(
                    self.voxel_layer.voxel_size)).int()
            
            self.get_voxel_seg(res_coors, data_sample)
            self.get_voxel_cand(res_coors, data_sample)
            
            center, fog = self.get_hm_bin(data_sample, polar_res_clamp.detach().cpu().numpy())     
            heatmaps.append(center)
            fogs.append(fog)
                
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            res_voxels = torch.cat((polar_res, res[:, :2], res[:, 3:]),
                                    dim=-1) # radial, angle, z, x, y, intensity
            voxels.append(res_voxels)
            coors.append(res_coors)
            
                
        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
       
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors
        voxel_dict['gt_heatmap_bev'] = heatmaps
        voxel_dict['gt_fog_bev'] = fogs

        return voxel_dict

    
    def get_voxel_seg(self, res_coors: torch.Tensor, data_sample: SampleList):
        """Get voxel-wise segmentation label and point2voxel map.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample: (:obj:`Det3DDataSample`): The annotation data of
                every samples. Add voxel-wise annotation forsegmentation.
        """

        if self.training:
            if hasattr(data_sample.gt_pts_seg, 'pts_instance_mask'):
                pts_instance_mask = data_sample.gt_pts_seg.pts_instance_mask
                pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
                num_points = res_coors.shape[0]

                _, unique_indices_inverse = torch.unique(
                    res_coors, return_inverse=True, dim=0)

                unq_instance_labels = torch.unique(pts_instance_mask, dim=0)
                sort_instance_labels = pts_instance_mask.new_zeros(pts_instance_mask.shape)

                i = 1
                ins2sem = pts_instance_mask.new_zeros(size=[len(unq_instance_labels)])
                for u in unq_instance_labels:
                    valid = pts_instance_mask == u
                    sort_instance_labels[valid] = i
                    ins2sem[i - 1] = pts_semantic_mask[valid][0]
                    i = i + 1

                flatten_inst_labels = res_coors.new_zeros(
                    (num_points, i))  # i: instance number+1(background)
                flatten_inst_labels[list(range(0, num_points)), # [num_points*one_hot_instance_labels]
                                    sort_instance_labels[list(range(0, num_points))].
                                    cpu().numpy().tolist()] += 1
                # scatter point-wise labels into voxel-wise
                flatten_inst_label_sum = torch_scatter.scatter_sum(
                    flatten_inst_labels, unique_indices_inverse, dim=0)
                voxel_instance_labels = torch.argmax(flatten_inst_label_sum, dim=1)
                voxel_semantic_labels = ins2sem[voxel_instance_labels - 1]
                data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_labels
                data_sample.gt_pts_seg.voxel_instance_mask = voxel_instance_labels

                
                # create point2voxel map for calculating voxel2point feature
                if self.return_points:
                    pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
                    voxel_len, s_voxel_coor, point2voxel_map = dynamic_scatter_3d(pseudo_tensor,
                                                        res_coors, 'mean', True)
                    data_sample.gt_pts_seg.point2voxel_map = point2voxel_map.long()
                    data_sample.gt_pts_seg.voxel_len = voxel_len
                    data_sample.gt_pts_seg.s_voxel_coor = s_voxel_coor

            else:
                # if instance mask not avaliable, 
                # use dynamic scatter to covert point-wise semantic mask into voxel-wise
                pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask 
                voxel_semantic_mask, _, point2voxel_map = dynamic_scatter_3d(
                    F.one_hot(pts_semantic_mask.long()).float(), res_coors, 'mean',
                    True)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_mask
                data_sample.gt_pts_seg.point2voxel_map = point2voxel_map

        else:
            # for validation and test mode, use dynamic scatter to covert into voxel-wise
            pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
            voxel_len, s_voxel_coor, point2voxel_map = dynamic_scatter_3d(pseudo_tensor,
                                                       res_coors, 'mean', True)
            data_sample.gt_pts_seg.point2voxel_map = point2voxel_map
            data_sample.gt_pts_seg.voxel_len = voxel_len
            
            data_sample.gt_pts_seg.s_voxel_coor = s_voxel_coor
            
            pts_instance_mask = data_sample.gt_pts_seg.pts_instance_mask
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            num_points = res_coors.shape[0]
            
            _, unique_indices_inverse = torch.unique(
                    res_coors, return_inverse=True, dim=0)

            unq_instance_labels = torch.unique(pts_instance_mask, dim=0)
            sort_instance_labels = pts_instance_mask.new_zeros(pts_instance_mask.shape)

            # map instance labels to [1, num_of_instances]
            i = 1
            ins2sem = pts_instance_mask.new_zeros(size=[len(unq_instance_labels)])
            for u in unq_instance_labels:
                valid = pts_instance_mask == u
                sort_instance_labels[valid] = i
                ins2sem[i - 1] = pts_semantic_mask[valid][0]
                i = i + 1

            flatten_inst_labels = res_coors.new_zeros(
                (num_points, i))  # i: instance number+1(background)
            flatten_inst_labels[list(range(0, num_points)), # [num_points*one_hot_instance_labels]
                                sort_instance_labels[list(range(0, num_points))].
                                cpu().numpy().tolist()] += 1
            # scatter point-wise labels into voxel-wise
            flatten_inst_label_sum = torch_scatter.scatter_sum(
                flatten_inst_labels, unique_indices_inverse, dim=0)
            voxel_instance_labels = torch.argmax(flatten_inst_label_sum, dim=1)
            voxel_semantic_labels = ins2sem[voxel_instance_labels - 1]
            
            data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_labels
            data_sample.gt_pts_seg.voxel_instance_mask = voxel_instance_labels
          
    def get_voxel_cand(self, res_coors, data_sample):
        sam_pmask_2d, sam_pscore_2d = data_sample.cand_2d['sam_pmask_2d'], data_sample.cand_2d['sam_pscore_2d']
        sam_pmask_2d = remap_instance_mask(torch.from_numpy(sam_pmask_2d).to(res_coors.device)).float() # remap the instance mask to 1, 2, 3, ..., and convert to float
        sam_pscore_2d = torch.from_numpy(sam_pscore_2d).to(res_coors.device)
        
        sam_vmask_2d, _, _ = dynamic_scatter_3d(sam_pmask_2d.unsqueeze(-1), res_coors, 'max', True)
        sam_vscore_2d_t, _, _ = dynamic_scatter_3d(sam_pscore_2d.unsqueeze(-1), res_coors, 'max', True)
        sam_vmask_2d[sam_vmask_2d<0] = 0 # avoid -inf in empty voxel
        sam_vscore_2d_t[sam_vmask_2d<0] = 0
        
        data_sample.cand_2d['sam_vmask_2d'] = sam_vmask_2d.squeeze(-1) 
        data_sample.cand_2d['sam_vscore_2d'] = sam_vscore_2d_t.squeeze(-1)
        data_sample.cand_2d['sam_pmask_2d'] = sam_pmask_2d.detach().cpu().numpy() 
        
    def get_hm_bin(self, data_sample, points):        
        grid_size = np.array(self.grid_shape)
        bev_grid_size = [self.grid_shape[0], self.grid_shape[1], 1]
        ignore_label = self.ignore_label
        point_cloud_range_polar = np.array(self.voxel_layer.point_cloud_range)
        min_bound, max_bound = point_cloud_range_polar[:3], point_cloud_range_polar[3:]
        intervals = (max_bound - min_bound) / grid_size
        
        plg = PanopticLabelGenerator(grid_size=grid_size, sigma=self.hm_sigma, polar=True)
        pts_instance_mask = data_sample.gt_pts_seg.pts_instance_mask.detach().cpu().numpy()
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask.detach().cpu().numpy()
        pts_instance_mask, ins2sem = resort_instance_labels(pts_instance_mask, pts_semantic_mask)
        
        mask = np.zeros_like(pts_semantic_mask, dtype=bool)
        for label in self.inst_labels:
            mask[np.logical_and(pts_semantic_mask == label, pts_instance_mask != 0)] = True
        
        grid_ind = (np.floor((np.clip(points, min_bound, max_bound) - min_bound) / intervals)).astype(int)
        current_grid = grid_ind 
        current_grid = np.clip(current_grid, 0, grid_size - 1)

        voxel_inst = pts_instance_mask[mask].squeeze()
        unique_inst = np.unique(voxel_inst)
        unique_inst_dict = {label: idx + 1 for idx, label in enumerate(unique_inst)}
        if voxel_inst.size > 1:
            voxel_inst = np.vectorize(unique_inst_dict.__getitem__)(voxel_inst)
            processed_inst = np.ones(grid_size[:2], dtype=np.uint8) * ignore_label
            inst_voxel_pair = np.concatenate([current_grid[mask, :2], voxel_inst[..., np.newaxis]], axis=1)
            inst_voxel_pair = inst_voxel_pair[np.lexsort((current_grid[mask, 0], current_grid[mask, 1])), :]
            processed_inst = nb_process_inst(np.copy(processed_inst), inst_voxel_pair)
        else:
            processed_inst = np.zeros([grid_size[0], grid_size[1]])
            
        dim_array = np.ones(len(grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(bev_grid_size) * intervals.reshape(dim_array) + np.array(min_bound).reshape(dim_array)

        center = plg(pts_instance_mask[mask], points[mask], 
                        processed_inst, voxel_position,
                        label_dict=unique_inst_dict,
                        min_bound=min_bound, intervals=intervals)
        instmap = processed_inst!=0 # [H, W]
        return center, instmap

def resort_instance_labels(pts_instance_mask, pts_semantic_mask):
    # map instance labels to [1, num_of_instances]
    sort_instance_labels = np.zeros_like(pts_instance_mask)

    unq_instance_labels = np.unique(pts_instance_mask)
    i = 1
    ins2sem = np.zeros(len(unq_instance_labels))
    for u in unq_instance_labels:
        valid = pts_instance_mask == u
        sort_instance_labels[valid] = i
        ins2sem[i - 1] = pts_semantic_mask[valid][0]
        i = i + 1
    return sort_instance_labels, ins2sem