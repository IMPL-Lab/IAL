import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba as nb
import matplotlib.pyplot as plt
import cv2
import torch
import pan_lib.cuda_kernel as cuda_kernel
import fpsample

class PanopticLabelGenerator(object):
    def __init__(self,grid_size,sigma=5,polar=False):
        """Initialize panoptic ground truth generator

        Args:
            grid_size: voxel size.
            sigma (int, optional):  Gaussian distribution paramter. Create heatmap in +-3*sigma window. Defaults to 5.
            polar (bool, optional): Is under polar coordinate. Defaults to False.
        """        
        self.grid_size = grid_size
        self.polar = polar

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    def __call__(self, inst, xyz, voxel_inst, voxel_position, label_dict, min_bound, intervals):
        """Generate instance center and offset ground truth

        Args:
            inst : instance panoptic label (N)
            xyz : point location (N x 3)
            voxel_inst : voxel panoptic label on the BEV (H x W)
            voxel_position : voxel location on the BEV (3 x H x W)
            label_dict : unqiue instance label dict
            min_bound : space minimal bound
            intervals : voxelization intervals

        Returns:
            center, center_pts, offset
        """        
        height, width = self.grid_size[0],self.grid_size[1]
        Z = self.grid_size[2]
        assert Z > 1, 'Height map requires Z > 1'
        
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        
        if inst.size < 2: return center
        inst_labels = np.unique(inst)
        for inst_label in inst_labels:
            mask = np.where(inst == inst_label)
            center_x, center_y = np.mean(xyz[mask,0]), np.mean(xyz[mask,1])
            x, y = int(np.floor((center_x-min_bound[0])/intervals[0])), int(np.floor((center_y-min_bound[1])/intervals[1]))
            center_pts.append([x, y])
            if x < 0 or y < 0 or \
                    x >= height or y >= width:
                x = np.clip(x, 0, height - 1)
                y = np.clip(y, 0, width - 1)
            sigma = self.sigma
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            if self.polar:
                c, d = max(0, -ul[0]), min(br[0], height) - ul[0]
                a, b = 0, br[1] - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], height)
                angle_list = [angle_id % width for angle_id in range(ul[1],br[1])]
                center[0, cc:dd, angle_list] = np.maximum(
                    center[0, cc:dd, angle_list], np.transpose(self.g[c:d,a:b]))

            else:
                c, d = max(0, -ul[0]), min(br[0], height) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], width) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], height)
                aa, bb = max(0, ul[1]), min(br[1], width)
                center[0, cc:dd, aa:bb] = np.maximum(
                    center[0, cc:dd, aa:bb], self.g[c:d,a:b])
        return center


@nb.jit('u1[:,:](u1[:,:],i8[:,:])', cache=True, parallel=False, nopython=True)
def nb_process_inst(processed_inst, sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_inst_voxel_pair[0, 2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0, :2]
    for i in range(1, sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i, :2]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i, 2]] += 1
    processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst

def bev2seq(bev_data, voxel_coor):
    """
    transform bev heatmap to sequence along the voxel coor
    params:
    - hm_bev: torch.Tensor, shape (H, W, C)
    - voxel_coor: torch.Tensor, shape (N, 2)
    return:
    - voxel_data: torch.Tensor, shape (N, C)
    """
    assert voxel_coor.shape[1] == 3, 'voxel_coor should be in shape (N, 3)'
    H, W, C = bev_data.shape
    indices = voxel_coor[:, 0] * W + voxel_coor[:, 1] 
    indices = indices.unsqueeze(1).expand(-1, C).long()

    voxel_data = torch.gather(bev_data.view(-1, C), 0, indices)
    return voxel_data

def normalize_heatmap(heatmap):
    "Normalize the heatmap by z-score and min-max normalization."
    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

def normalize_fog(fog):
    """Normalize the fog by sigmoid and min-max normalization."""
    if not isinstance(fog, torch.Tensor):
        fog = torch.tensor(fog, dtype=torch.float32)
    if fog.size(0) == 1 and len(fog.size()) == 3:
        fog = fog.squeeze(0)
        
    fog = torch.sigmoid(fog) 
    fog = (fog - fog.min()) / (fog.max() - fog.min()) 
    return fog

def extract_selected_features(candidates, voxel_feat):
    """
    extract features from top N activations
    Args:
        - candidates: Tensor, shape (N), top N activations
        - voxel_feat: Tensor, shape (V, C), voxel features
    Returns:
        - selected_feat: Tensor, shape (N, C), selected features
    """
    selected_feat = torch.gather(voxel_feat, 0, candidates.unsqueeze(1).expand(-1, voxel_feat.shape[1]))
    return selected_feat
def nms_polar(heatmap, nms_kernel=3):
    """
    Non-maximum suppression on polar heatmap.
    """    
    ctr_hmp = heatmap.clone().unsqueeze(0).unsqueeze(0)
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.pad(ctr_hmp,(nms_padding,nms_padding,0,0),mode = 'circular')
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp_max_pooled, kernel_size=nms_kernel, stride=1, padding=(nms_padding,0))
    if ctr_hmp_max_pooled.shape != ctr_hmp.shape:
        ctr_hmp_max_pooled = F.interpolate(ctr_hmp_max_pooled, size=(heatmap.shape[1], heatmap.shape[2]), mode='nearest')
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = 0
    return ctr_hmp.squeeze(0)

def select_coor_score(heatmap, voxel_coor, index):
    if voxel_coor.dtype != torch.long:
        voxel_coor = voxel_coor.long()
    if index.dtype != torch.long:
        index = index.long()
    cand_vcoor = voxel_coor[index]
    if len(cand_vcoor.shape) == 1:
        cand_vcoor = cand_vcoor.unsqueeze(0)
        
    cand_score = heatmap[0, cand_vcoor[:, 0], cand_vcoor[:, 1]]
    return cand_vcoor, cand_score

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def opencv_dilate(points, kernel_size, heatmap_shape):
    heatmap = np.zeros(heatmap_shape, dtype=np.uint8)
    for point in points:
        x, y = point
        heatmap[x, y] = 1

    kernel = gaussian_2d([kernel_size, kernel_size], sigma=1) # gaussian kernel
    heatmap = cv2.dilate(heatmap, kernel, iterations=1)
    return heatmap

def select_center_voxel(heatmap, voxel_coors, nms_func, num_cand=128, k=5, filter_with_thre=True):
    """ Find the center of the heatmap in BEV space, search nearby for the missing voxel coor."""
    
    if len(heatmap.shape) == 3: # compress the class dimension
        heatmap = heatmap.max(0)[0]
    if len(heatmap.shape) == 2:
        heatmap = heatmap
        
    heatmap = normalize_heatmap(heatmap)
    thre = heatmap.mean()
    heatmap[heatmap < heatmap.mean()] = 0
    pred_nms = nms_polar(heatmap, k).squeeze(0)
    ctr_all = torch.nonzero(pred_nms > thre) if filter_with_thre else torch.nonzero(pred_nms)

    if ctr_all.size(0) >= num_cand*2:
        top_k_scores, _ = torch.topk(pred_nms.flatten(), num_cand*2)
        selected = torch.nonzero(pred_nms >= top_k_scores[-1])
    else:
        selected = ctr_all
    
    dil_heatmap = opencv_dilate(selected.cpu().numpy(), k, (480, 360)) # for-loop:0.002s # cv2: 0.0002s
    dil_heatmap = torch.tensor(dil_heatmap).to(voxel_coors.device).float() if not isinstance(dil_heatmap, torch.Tensor) else dil_heatmap
    voxel_scores = bev2seq((heatmap*dil_heatmap).unsqueeze(-1), voxel_coors).squeeze(-1)
    selected_coor, _, selected_ind = nms_func(voxel_coors, voxel_scores)
    selected_coor = selected_coor.long()
    return None, selected_ind

def select_3DQ(heatmap, voxel_coors_list, nms_para):
    point_nms = BEVPointNMS(nms_para)
    norm_heatmap = normalize_heatmap(heatmap)
    _, ind_3d = select_center_voxel(norm_heatmap, voxel_coors_list, point_nms, 
                                               k=5, filter_with_thre=True)
    
    cand_vcoor_3d, cand_vscore_3d = select_coor_score(norm_heatmap, voxel_coors_list, ind_3d)
    ind_3d = ind_3d.to(voxel_coors_list.device)
    return cand_vcoor_3d.long(), cand_vscore_3d, ind_3d

def select_2DQ(voxel_coor, sam_vmask_2d, sam_vscore_2d, pred_fog, grid_size=[480, 360, 32]):
    """
    Select 2D queries based on 2D candidate mask and score. Use fog and distance as additional factors.
    Parameters:
        data_sample (dict): data sample.
        voxel_coor (tensor): voxel coordinates.
    Returns:
        cand_vcoor_2d (tensor): selected 2D query coordinates.
        cand_vscore_2d (tensor): selected 2D query scores.
    """
    ind_2d = torch.arange(0, sam_vmask_2d.shape[0], device=sam_vmask_2d.device)
    fog_mask_bev = normalize_fog(pred_fog)
    fog_mask_seq = bev2seq(fog_mask_bev.squeeze(0).unsqueeze(-1), voxel_coor).squeeze(-1)
    sam_vscore_2d = sam_vscore_2d*fog_mask_seq
    dist_s = voxel_coor[:, 0]/grid_size[0]
    sam_vscore_2d = sam_vscore_2d * dist_s

    cand_vmask_ctr_2d = extract_pointset_centers_torch(sam_vmask_2d, voxel_coor.float())
    cand_vcoor_2d = voxel_coor[cand_vmask_ctr_2d>0,:] 
    cand_vscore_2d = sam_vscore_2d[cand_vmask_ctr_2d>0]
    ind_2d = ind_2d[cand_vmask_ctr_2d>0]
    return cand_vcoor_2d, cand_vscore_2d, ind_2d

def cand_recall_calc(gt_inst, pred_cand, thre=0):
    """
    Calculate the recall of instance segmentation.
    
    Args:
    - gt_inst: Tensor, shape (N), the ground truth instance label for each point/voxel.
    - pred_cand: Tensor, shape (N), the predicted candidate instance, only contains the mask of some instances.
    - thre: float, IoU threshold, the threshold to judge whether the candidate instance is correctly predicted.
    
    Returns:
    - recall: float, the recall of candidate instance.
    """
    if not isinstance(gt_inst, torch.Tensor):
        gt_inst = torch.tensor(gt_inst)
    if not isinstance(pred_cand, torch.Tensor):
        pred_cand = torch.tensor(pred_cand)
    
    # Re-index gt_inst to ensure that the instances are numbered starting from 0
    unique_gt, gt_inst_reindexed = torch.unique(gt_inst, return_inverse=True)
    M = unique_gt.shape[0]  # Number of real instances

    # One-hot encoding for gt_inst
    gt_inst_onehot = torch.zeros((gt_inst.shape[0], M), device=gt_inst.device)
    gt_inst_onehot.scatter_(1, gt_inst_reindexed.unsqueeze(1), 1)

    # Calculate the overlap 
    overlaps = torch.einsum('pm,p->pm', gt_inst_onehot, pred_cand.float())
    res = overlaps.max(dim=0)[0] > thre

    return res

def extract_pointset_centers_torch(pts_mask, pts_coor):
    unique_sets = torch.unique(pts_mask)
    pts_center = torch.zeros_like(pts_coor)
    
    for pointset_id in unique_sets:
        mask = pts_mask == pointset_id
        pointset_coor = pts_coor[mask.squeeze()]
        
        centroid = pointset_coor.mean(dim=0)
        distances = torch.norm(pointset_coor - centroid, dim=1)
        nearest_point_index = torch.argmin(distances)
        pts_center[mask.nonzero(as_tuple=True)[0][nearest_point_index]] = 1
        pts_center = pts_center.max(1)[0].unsqueeze(1)
    return pts_center.squeeze(-1)

class BEVPointNMS(nn.Module):
    def __init__(self, point_nms_dic):
        super(BEVPointNMS, self).__init__()
        self.point_nms_dic = point_nms_dic
        self.K = self.point_nms_dic['K']
        self.dist_thresh = float(self.point_nms_dic['dist_thresh'])
        self.vote_thresh = float(self.point_nms_dic['vote_thresh'])
        self.score_thresh = float(self.point_nms_dic['score_thresh'])
        if 'grid_size' in self.point_nms_dic.keys():
            self.grid_size = self.point_nms_dic['grid_size']
        if 'pts_range_polar' in self.point_nms_dic.keys():
            self.pts_range_polar = self.point_nms_dic['pts_range_polar']
        if 'pts_range_cart' in self.point_nms_dic.keys():
            self.pts_range_cart = self.point_nms_dic['pts_range_cart']
            
    def forward(self, points, scores):
        if points.dtype != torch.float:
            points = points.float()
        if scores.dtype != torch.float:
            scores = scores.float()
            
        DEVICE = points.device
        pts_score = torch.cat((points, scores.unsqueeze(-1)), dim=-1)
        select_ind = torch.arange(pts_score.shape[0])

        N = pts_score.shape[0]
        _, indices = pts_score[:, -1].sort(dim=0, descending=True)
        select_ind_order = select_ind[indices]
        pts_score_order = pts_score[indices]
        select_ind_order = select_ind_order[pts_score_order[:, -1] > self.score_thresh]
        pts_score_order = pts_score_order[pts_score_order[:, -1] > self.score_thresh]
         
        col_blocks = (N + 64 - 1) // 64
        obj_center = torch.zeros((self.K, 3), dtype=torch.float).to(DEVICE)
        matching_mat = torch.zeros((N, col_blocks), dtype=torch.long).to(DEVICE)
        matching_mat_vote = torch.zeros((N, col_blocks), dtype=torch.long).to(DEVICE)
        remv = torch.zeros((col_blocks,), dtype=torch.long).to(DEVICE)
        keep_order = torch.zeros((self.K,), dtype=torch.long).to(DEVICE)

        if not isinstance(self.grid_size, torch.Tensor):
            self.grid_size = torch.tensor(self.grid_size).to(DEVICE)
        self.pts_range_polar = torch.tensor(self.pts_range_polar).to(DEVICE)
        self.pts_range_cart = torch.tensor(self.pts_range_cart).to(DEVICE)
        
        bev_order = pts_score_order[:, :3]/self.grid_size
        bev_order = cartisianize_norm(bev_order, self.pts_range_polar, self.pts_range_cart, return_normed=False)
        bev_order[:, 2] = torch.zeros_like(bev_order[:, 2])
        bev_score_order = torch.cat([bev_order, pts_score_order[:, -1].unsqueeze(-1)], dim=1)
        cuda_kernel.vote_nms_fast_gpu(bev_score_order, obj_center, 
                                    matching_mat, matching_mat_vote, remv, keep_order, 
                                    self.dist_thresh, self.vote_thresh)

        keep_order = keep_order[keep_order > -1]  
        return pts_score_order[keep_order][:, :3], obj_center, select_ind_order[keep_order]
    
    
def cartisianize_norm(normed_polar_coor, point_cloud_range_polar, point_cloud_range_cart, return_normed=True):
    """
    Convert normalized polar coordinates to cartisian coordinates.
    Params:
        normed_polar_coors: [N, 3]
        point_cloud_range_polar: polar range of point cloud, [rad_min, ang_min, z_min, rad_max, ang_max, z_max]
        point_cloud_range_cart: cartisian range of point cloud, [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        normed_cat_coors: [N, 3]
    """        
    point_cloud_range_polar = torch.tensor(point_cloud_range_polar, dtype=torch.float).to(normed_polar_coor.device) if isinstance(point_cloud_range_polar, list) else point_cloud_range_polar
    point_cloud_range_cart = torch.tensor(point_cloud_range_cart, dtype=torch.float).to(normed_polar_coor.device) if isinstance(point_cloud_range_cart, list) else point_cloud_range_cart

    polar_coor = normed_polar_coor.new_zeros(normed_polar_coor.shape)
    for i in range(3):
        polar_coor[:, i] = normed_polar_coor[:, i]*(
                            point_cloud_range_polar[i+3] -
                            point_cloud_range_polar[i]) + \
                            point_cloud_range_polar[i]
    x = polar_coor[:, 0] * torch.cos(polar_coor[:, 1])
    y = polar_coor[:, 0] * torch.sin(polar_coor[:, 1])
    cat_coor = torch.stack([x, y, polar_coor[:, 2]], 1)
    
    if not return_normed:
        return cat_coor
    else:
        normed_xy = (cat_coor[:, :2] - point_cloud_range_cart[:2]) / (point_cloud_range_cart[3:5] - point_cloud_range_cart[:2])

        height = (cat_coor[:, 2] - point_cloud_range_cart[2]) / (point_cloud_range_cart[5] - point_cloud_range_cart[2])
        normed_cart_coor = torch.cat([normed_xy, height[:, None]], 1)
        return normed_cart_coor
    
def fps_sample(pc, num_samples):    
    if isinstance(pc, torch.Tensor):
        pc_input = pc.cpu().numpy().astype(np.float32)
    else:
        pc_input = pc.copy().astype(np.float32)
    if pc_input.shape[0] <= num_samples:
        return pc, torch.arange(pc_input.shape[0]).to(pc.device).long()
    num_samples = int(num_samples)
    idx = fpsample.bucket_fps_kdline_sampling(pc_input, num_samples, h=3)
    if isinstance(pc, torch.Tensor):
        idx = torch.from_numpy(idx.astype(np.int64)).to(pc.device).long()
    return pc[idx], idx