import numpy as np
import torch
from typing import Tuple
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

import os.path as osp
from PIL import Image

class _LidarPointCloud(LidarPointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def pack_points(cls, points: str) -> '_LidarPointCloud':
        """
        Loads LIDAR data from point cloud format. Data is stored as (x, y, z, intensity, ring index) with shape of (N, 4).
        :param points: data of point cloud.
        :return: LidarPointCloud instance (x, y, z, intensity) with shape of (4, N).
        """
        return cls(points.T)
    
def proj_lidar2img_nusc(nusc,
                            points_data: np.ndarray,  # (4, N)
                            point_calibrated_sensor_token: str,
                            point_ego_pose_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            return_img: bool = False,
                            aug_mat: np.ndarray = None,
                            sem_mask: np.ndarray = None,
                            use_label_map: bool = False
                            ) -> Tuple:
    """
    Given a point cloud and camera sample_data token(view info), map it to the respective image plane.
    :param points_data: Lidar data with shape of (N,4).
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :return (mapped coordinate <np.float: 3, n)>, coloring(optional) <np.float: n>, image(optional) <Image>).
    """
    if sem_mask is not None:
        if np.unique(sem_mask).shape[0]<=2:
            color_map = {
                0: [173, 197, 207], # light lake blue
                1: [41, 96, 115], # lake blue
            }
        else:
            color_map = {
                0: [0, 0, 0],  # noise                 black
                1: [255, 120, 50],  # barrier               orange
                2: [255, 192, 203],  # bicycle               pink
                3: [255, 255, 0],  # bus                   yellow
                4: [0, 150, 245],  # car                   blue
                5: [0, 255, 255],  # construction_vehicle  cyan
                6: [255, 127, 0],  # motorcycle            dark orange
                7: [255, 0, 0],  # pedestrian            red
                8: [255, 240, 150],  # traffic_cone          light yellow
                9: [135, 60, 0],  # trailer               brown
                10: [160, 32, 240],  # truck                 purple
                11: [255, 0, 255],  # driveable_surface     dark pink
                12: [139, 137, 137],  # other_flat            dark red
                13: [75, 0, 75],  # sidewalk              dark purple
                14: [150, 240, 80],  # terrain               light green
                15: [230, 230, 250],  # manmade               white
                16: [0, 175, 0],  # vegetation            green
            }   
        label_map = {
            1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0,
            9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7, 12: 8,
            22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16
            }
    
    img_H, img_W = 900, 1600 
    cam = nusc.get('sample_data', camera_token) 
    pc = _LidarPointCloud.pack_points(points_data.copy())
        
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # Init step: augment transformation: rotate, scale, translate
    if aug_mat is not None:
        aug_mat_inv = np.linalg.inv(aug_mat)    
        pc.rotate(aug_mat[:3, :3].T)
        
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', point_calibrated_sensor_token)
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
        
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', point_ego_pose_token)
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    
    depths = pc.points[2, :]
    if return_img:
        im = Image.open(osp.join(nusc.dataroot, cam['filename']))

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        # coloring is depth, other visualizations are possible if necessary
        if sem_mask is not None:
            coloring = sem_mask
            # map semantic label to color
            if use_label_map:
                coloring = np.array([color_map[label_map[i]] for i in coloring])
            else:
                coloring = np.array([color_map[i] for i in coloring])
            
            coloring = coloring/255
        else:
            coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img_W - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img_H - 1)
        
    if return_img:
        return points, coloring, im, mask
    else:
        return points, mask

def proj_lidar2img(points, lidar2img, img_size=(1600, 900), min_dist=1.0):
    """Project lidar points to image plane.
    Args:
        points (torch.Tensor): Lidar points. (N, 3)
        lidar2img (torch.Tensor): Lidar to image matrix. (4, 4)
        img_size (tuple): Image size.
        min_dist (float): Minimum distance to the camera.
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(lidar2img, np.ndarray):
        lidar2img = torch.from_numpy(lidar2img).float()
    
    N = points.shape[0]
    device = points.device
    if N == 0:
        return torch.empty((0, 2), device=device), torch.zeros(0, dtype=torch.bool, device=device)

    points = torch.cat([points, torch.ones(points.shape[0],1).to(points)], dim=1)
    lidar2img = lidar2img.to(points)
    points_img = (lidar2img @ points.T)
    depths = points_img[2, :]
    points_img = points_img / points_img[2]
    img_W, img_H = img_size
    
    mask = torch.ones(depths.shape[0], dtype=torch.bool).to(points.device)
    mask = torch.logical_and(mask, depths > min_dist)
    mask = torch.logical_and(mask, points_img[0, :] > 1)
    mask = torch.logical_and(mask, points_img[0, :] < img_W - 1)
    mask = torch.logical_and(mask, points_img[1, :] > 1)
    mask = torch.logical_and(mask, points_img[1, :] < img_H - 1)

    points_img = points_img[:, mask]
    return points_img[:2, :].T, mask

def polar2cartesian(polar_coor, aux_feat=True):
    x = polar_coor[:, 0] * np.cos(polar_coor[:, 1])
    y = polar_coor[:, 0] * np.sin(polar_coor[:, 1]) 
    z = polar_coor[:, 2] 
    if aux_feat:
        return np.stack([x, y, z, polar_coor[:, -1]], axis=1) # return coord should be in (N_V, 4)
    else:
        return np.stack([x, y, z], axis=1)

def cartesian2polar(cartesian_coor, aux_feat=True):
    x = cartesian_coor[:, 0]
    y = cartesian_coor[:, 1]
    z = cartesian_coor[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    if aux_feat:
        return np.stack([r, theta, z, cartesian_coor[:, -1]], axis=1) # return coord should be in (N_V, 4)
    else:
        return np.stack([r, theta, z], axis=1)
    
def update_lidars2imgs_info(img_meta, dataset_type='nuscenes'):
    if dataset_type == 'nuscenes':
        assert 'lidar2cam' in img_meta.keys()
        assert 'cam2img' in img_meta.keys()
        
        if isinstance(img_meta['lidar2cam'][0], list):
            lidars2cams = np.array(img_meta['lidar2cam'])
            lidars2cams = torch.tensor(lidars2cams).float()
        elif isinstance(img_meta['lidar2cam'][0], torch.Tensor):
            lidars2cams = torch.stack(img_meta['lidar2cam'], dim=0)
        else:
            raise TypeError(f'Unsupported data type for lidar2cam as {img_meta["lidar2cam"][0].dtype}, should be list or torch.Tensor')

        if isinstance(img_meta['cam2img'][0], list):
            cams2imgs = np.array(img_meta['cam2img'])
            cams2imgs = torch.tensor(cams2imgs).float()
        elif isinstance(img_meta['cam2img'][0], torch.Tensor):
            cams2imgs = torch.stack(img_meta['cam2img'], dim=0)
        else:
            raise TypeError(f'Unsupported data type for cam2img as {img_meta["cam2img"][0].dtype}, should be list or torch.Tensor')

        lidars2imgs, imgs2lidars = lidar2imgs_batched(lidars2cams, cams2imgs)
        img_meta['lidars2imgs_torch'] = lidars2imgs
        img_meta['imgs2lidars_torch'] = imgs2lidars

    elif dataset_type == 'semantickitti':  # Consider using lidars2imgs and imgs2lidar instead of P and Tr matrics
        assert 'p_dict' in img_meta and 'tr_mat' in img_meta
        img_meta['p_torch'] = {
            'left': torch.from_numpy(img_meta['p_dict']['left']).float(),
            'right': torch.from_numpy(img_meta['p_dict']['right']).float()}
        img_meta['tr_torch'] = torch.from_numpy(img_meta['tr_mat']).float()
    return img_meta

def pack_lidar2imgs(lidar2cams, cam2imgs):
    lidar2imgs = []
    for view in range(len(lidar2cams)):
        lidar2cam, cam2img = lidar2cams[view], cam2imgs[view]
        lidar2cam = np.array(lidar2cam) if not isinstance(lidar2cam, np.ndarray) else lidar2cam
        cam2img = np.array(cam2img) if not isinstance(cam2img, np.ndarray) else cam2img
        lidar2img = lidar2img_single(lidar2cam,  cam2img)
        lidar2imgs.append(lidar2img)
    return lidar2imgs

def lidar2img_single(lidar2cam, cam2img):
    assert lidar2cam.shape == (4, 4)   
    assert cam2img.shape == (3, 3)
    
    cam2img_pad = torch.eye(4) if isinstance(lidar2cam, torch.Tensor) else np.eye(4)
    cam2img_pad[:3, :3] = cam2img
    if isinstance(lidar2cam, np.ndarray):
        lidar2img = np.matmul(cam2img_pad, lidar2cam)
    elif isinstance(lidar2cam, torch.Tensor):
        lidar2img = torch.matmul(cam2img_pad, lidar2cam) 
    else:
        raise TypeError(f'Unsupported data type for lidar2cam as {lidar2cam.dtype}, should be np.ndarray or torch.Tensor') 
    return lidar2img

def lidar2imgs_batched(lidars2cams, cams2imgs):
    assert lidars2cams.shape[1:] == (4, 4)   
    assert cams2imgs.shape[1:] == (3, 3)
    N_view = lidars2cams.shape[0]
    cams2imgs_pad = torch.zeros(N_view, 4, 4)
    cams2imgs_pad[:, :3, :3] = cams2imgs
    cams2imgs_pad[:, 3, 3] = 1

    lidars2cams = torch.bmm(cams2imgs_pad, lidars2cams)
    imgs2lidars = torch.inverse(lidars2cams)

    return lidars2cams, imgs2lidars

