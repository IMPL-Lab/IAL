import os
import torch
import numpy as np
from mmcv.ops import SparseConvTensor

def load_viewimages_path(pklfile, data_root):
    info_list = pklfile['data_list']
    VIEWS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    data_prefix = {
        'dataset_path': data_root,
        'lidar_path': data_root+'/samples/LIDAR_TOP',
        'pts_panoptic_mask': data_root+'/panoptic/v1.0-mini',
        'CAM_FRONT': data_root+'/samples/CAM_FRONT',
        'CAM_FRONT_RIGHT': data_root+'/samples/CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT': data_root+'/samples/CAM_FRONT_LEFT',
        'CAM_BACK': data_root+'/samples/CAM_BACK',
        'CAM_BACK_LEFT': data_root+'/samples/CAM_BACK_LEFT',
        'CAM_BACK_RIGHT': data_root+'/samples/CAM_BACK_RIGHT',}

    res_list = []
    for info in info_list:
        res = {}
        # load lidar points
        lidar_prefix = data_prefix.get('lidar_path')
        lidar_path = info['lidar_points']['lidar_path']
        lidar_path = os.path.join(lidar_prefix, lidar_path)
        res['lidar_path'] = lidar_path
        
        # load multi-view images
        img_files_path = {}
        for cam_id, img_info in info['images'].items(): # add prefix to img_path
            if 'img_path' in img_info:
                if cam_id in data_prefix:
                    cam_prefix = data_prefix[cam_id]
                img_info['img_path'] = os.path.join(cam_prefix,
                                                img_info['img_path'])
                img_files_path[cam_id] = img_info['img_path']
        
        meta_info = {}
        meta_info['imgs'] = {}
        meta_info['img_path'] = {}
        for view in VIEWS:
            meta_info['img_path'][view] = img_files_path[view]
        res['imgs_meta'] = meta_info
        res_list.append(res)
        res['sample_token'] = info['token']
        
        # load label
        dataset_path = data_prefix.get('dataset_path')
        panoptic_path = info['pts_panoptic_mask_path']
        panoptic_path = os.path.join(dataset_path, panoptic_path)
        res['label_path'] = panoptic_path
        
    return res_list
def load_point(pts_filename, backend_args=None, load_dim=5, use_dim=range(3)):
    try:
        from mmengine.fileio import get
        pts_bytes = get(pts_filename, backend_args=backend_args)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
    except:
        points = np.fromfile(pts_filename, dtype=np.float32)
    points = points.reshape(-1, load_dim)
    points = points[:, use_dim]
    return points

def load_mask(pts_panoptic_mask_path, seg_3d_dtype=np.int64, seg_offset=1000):
    pts_panoptic_mask = np.load(pts_panoptic_mask_path) 
    
    pts_panoptic_mask = pts_panoptic_mask['data']
    pts_semantic_mask = pts_panoptic_mask // seg_offset
    pts_instance_mask = pts_panoptic_mask.astype(seg_3d_dtype)
    return pts_semantic_mask, pts_instance_mask

def remap_instance_mask(inst_mask):
    unique_vals, inverse_indices = torch.unique(inst_mask, return_inverse=True)
    remapped_mask = inverse_indices.view(inst_mask.shape)
    return remapped_mask

def clone_detach_SparseConvTensor(sparse_tensor):
    features_copy = sparse_tensor.features.clone().detach()
    indices_copy = sparse_tensor.indices.clone().detach()

    sparse_tensor_copy = SparseConvTensor(
    features_copy,
    indices_copy,
    sparse_tensor.spatial_shape,
    sparse_tensor.batch_size
    )
    return sparse_tensor_copy

def expand_mask_torch(a: torch.Tensor, scale_factor=1/0.4, dim_expand=False,
                      ignore_ids=(0,),
                      dtype=torch.bool):
    device = a.device

    if scale_factor != 1.0:
        if dim_expand:
            H, W = a.shape
        else:
            N, H, W = a.shape
        Hp = max(1, int(round(H * float(scale_factor))))
        Wp = max(1, int(round(W * float(scale_factor))))
        if dim_expand:
            size = (Hp, Wp)
        else:
            size = (N, Hp, Wp)
        a_scaled = F.interpolate(
            a.unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=size, mode='nearest'
        ).squeeze(0).squeeze(0).to(a.dtype)
    else:
        a_scaled = a
        Hp, Wp = H, W
    
    if dim_expand:
        ids = torch.unique(a_scaled)
        if ignore_ids is not None and len(ignore_ids) > 0:
            ignore_ids_t = torch.tensor(ignore_ids, device=device, dtype=ids.dtype)
            keep = ~(ids.view(-1, 1) == ignore_ids_t.view(1, -1)).any(dim=1)
            ids = ids[keep]

        if ids.numel() == 0:
            empty = torch.empty(0, Hp, Wp, device=device, dtype=dtype)
            return empty, ids

        b = (ids.view(-1, 1, 1) == a_scaled.unsqueeze(0))  # bool

        if dtype is not torch.bool:
            b = b.to(dtype)
    else:
        b = a_scaled.to(dtype)

    return b