import torch

def get_center_scale_voxel(voxel_coor, grid_size, polar_range, cart_range, ret_scale=False):
    assert voxel_coor.shape[1]==3, 'The shape of voxel_coor should be [N, 3].'
    if not isinstance(grid_size, torch.Tensor):
        grid_size = torch.tensor(grid_size).to(voxel_coor.device).float()
    
    centeroid_coor = cylinder_center_index(voxel_coor)
    norm_polar_cent = centeroid_coor / grid_size
    norm_cart_cent  = cartisianize(norm_polar_cent, polar_range, cart_range)
    if not ret_scale:
        return norm_polar_cent, norm_cart_cent
    
    corner = find_corners(voxel_coor, grid_size) 
    norm_polar_corner = corner / (grid_size[None, None, :].float()-0)
    norm_cart_corner = cartisianize(norm_polar_corner.view(-1, 3), polar_range, cart_range).view(8, -1, 3) 
    scale = torch.sqrt(torch.sum(norm_cart_corner - norm_cart_cent.unsqueeze(0).expand_as(norm_cart_corner)**2, dim=2)).unsqueeze(2) 
    return norm_polar_cent, norm_cart_cent, scale

def cartisianize(normed_polar_coor, point_cloud_range_polar, point_cloud_range_cart):      
    point_cloud_range_polar = torch.tensor(point_cloud_range_polar, dtype=torch.float).to(normed_polar_coor.device)
    point_cloud_range_cart = torch.tensor(point_cloud_range_cart, dtype=torch.float).to(normed_polar_coor.device)
    
    polar_coor = normed_polar_coor.new_zeros(normed_polar_coor.shape)
    for i in range(3):
        polar_coor[:, i] = normed_polar_coor[:, i]*(
                            point_cloud_range_polar[i+3] -
                            point_cloud_range_polar[i]) + \
                            point_cloud_range_polar[i]
    x = polar_coor[:, 0] * torch.cos(polar_coor[:, 1])
    y = polar_coor[:, 0] * torch.sin(polar_coor[:, 1])
    cat_coor = torch.stack([x, y, polar_coor[:, 2]], 1)
    normed_xy = (cat_coor[:, :2] - point_cloud_range_cart[:2]) / (point_cloud_range_cart[3:5] - point_cloud_range_cart[:2])

    height = (cat_coor[:, 2] - point_cloud_range_cart[2]) / (point_cloud_range_cart[5] - point_cloud_range_cart[2])
    normed_cart_coor = torch.cat([normed_xy, height[:, None]], 1)
    return normed_cart_coor

def cylinder_center_index(voxel_coors):
    if not isinstance(voxel_coors, torch.Tensor):
        voxel_coors = torch.tensor(voxel_coors, dtype=torch.int)
    voxel_centers = voxel_coors.float() + 0.5
    return voxel_centers

def find_corners(voxel_coor, grid_size):
    radius_div, angle_div, height_div = grid_size
    
    corner_offsets = torch.tensor([
    [-1, -1, -1], # left front bottom 
    [-1,  1, -1], # left back bottom 
    [ 1, -1, -1], # right front bottom 
    [ 1,  1, -1], # right back bottom 
    [-1, -1,  1], # left front upper 
    [-1,  1,  1], # left back upper 
    [ 1, -1,  1], # right front upper 
    [ 1,  1,  1], # right back upper 
    ], dtype=torch.long)
    corner_offsets = (corner_offsets/2.).to(voxel_coor.device)

    corners = voxel_coor.expand(8, -1, -1)+corner_offsets.unsqueeze(1)+0.5

    # Ensuring that the corner indices are within the valid range
    corners = torch.stack([
    corners[..., 0].clamp(0, radius_div - 1),
    corners[..., 1].clamp(0, angle_div - 1),
    corners[..., 2].clamp(0, height_div - 1)
    ], dim=-1)

    return corners