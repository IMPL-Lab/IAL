import cv2
import torch
import numpy as np
import random

from tools.projection import proj_lidar2img

def fit_to_box(points, mode):
    box = fit_box_cv(points)
    if   mode == 'vertical':   k = 0
    elif mode == 'horizontal': k = 1
    else: raise ValueError('mode should be vertical or horizontal')
    
    sorted_by_x = sorted(box, key=lambda p: p[k]) 
    side1 = sorted_by_x[:2]  
    side1_mid = np.mean(side1, axis=0)  

    side2 = sorted_by_x[-2:] 
    side2_mid = np.mean(side2, axis=0) 
    return side1_mid[k].astype(int), side2_mid[k].astype(int)

def fit_box_cv(points):
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box

def merge_images_yaw(im1, im2, l, r):
    """merge the mixed image to the right side of the original image"""
    assert im1.shape == im2.shape
    assert l <= r
    
    l, r = int(l), int(r)
    W = im1.shape[1]
    
    new = np.array(im2).copy()
    orig = np.array(im1).copy()
    
    new[:, l:r, :] = orig[:, l:r, :]
    return new


def expand_box(box, x_expand, image_shape):
    img_height, img_width = image_shape[:2]
    x_coords = box[:, 0]
    y_coords = box[:, 1]

    min_x = max(np.min(x_coords) - x_expand, 0)  
    max_x = min(np.max(x_coords) + x_expand, img_width - 1)  
    min_y = max(np.min(y_coords) - x_expand, 0) 
    max_y = min(np.max(y_coords) + x_expand, img_height - 1)  

    expanded_box = np.array([[min_x, min_y],
                             [max_x, min_y],
                             [max_x, max_y],
                             [min_x, max_y]], dtype=np.int32)
    
    return expanded_box


def crop_box_img(box, img):
    assert isinstance(img, np.ndarray)
    assert box.min() >= 0
    x, y, w, h = cv2.boundingRect(box)
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def paste_box_img(box, source_img, target_img):
    assert isinstance(source_img, np.ndarray)
    assert isinstance(target_img, np.ndarray)
    assert box.min() >= 0
    x, y, w, h = cv2.boundingRect(box)
    if not source_img[y:y+h, x:x+w].shape == target_img.shape:
        target_img = cv2.resize(target_img, (w, h))
    source_img[y:y+h, x:x+w] = target_img
    return source_img

def merge_images_pitch_torch(im1, im2, point_block):
    """merge two images with pitch angle according to point block"""
    assert point_block.shape[1] == 2, 'point block should be (n, 2)'
    if isinstance(im1, np.ndarray):
        im1 = torch.from_numpy(im1).to(point_block.device)
    if isinstance(im2, np.ndarray):
        im2 = torch.from_numpy(im2).to(point_block.device)        
    t, b = fit_to_box(point_block.cpu().numpy(), 'horizontal')
    orig = im1.clone()
    new = im2.clone()
    mask = torch.zeros_like(orig)
    orig[t:b,:,:] = new[t:b,:,:]
    mask[t:b,:,:] = 1
    return orig, mask

def merge_images_yaw_torch(im1, im2, point_block):
    """merge two images with yaw angle according to point block"""
    assert point_block.shape[1] == 2, 'point block should be (n, 2)'
    if isinstance(im1, np.ndarray):
        im1 = torch.from_numpy(im1).to(point_block.device)
    if isinstance(im2, np.ndarray):
        im2 = torch.from_numpy(im2).to(point_block.device)        
    l,r = fit_to_box(point_block.cpu().numpy(), 'vertical')
    if l == r: return im1, torch.zeros_like(im1)
    l = max(0, l)
    r = min(im1.shape[1], r)
    
    orig = im1.clone()
    new = im2.clone()
    
    mask = torch.zeros_like(orig)
    orig[:,l:r,:] = new[:,l:r,:]
    mask[:,l:r,:] = 1
    
    return orig, mask


def select_mode_by_ratio(ratio):
    """
    Select a mode (H or V) based on the given ratio.

    Args:
        ratio (list): A list containing the weights for H and V, e.g., [1, 2].

    Returns:
        str: The selected mode, either 'H' or 'V'.
    """
    if len(ratio) != 2 or any(r < 0 for r in ratio):
        raise ValueError("Ratio must be a list of two non-negative numbers, e.g., [1, 2].")
    
    total = sum(ratio)
    prob_H = ratio[0] / total
    prob_V = ratio[1] / total
    selected = random.choices(['H', 'V'], weights=[prob_H, prob_V], k=1)[0]
    return selected


def divide_point_cloud_yaw(n):
    """
    Divide the yaw angle range [-pi, pi] into N equal parts with a random starting angle.

    Returns:
        list: A list of N tuples representing the yaw angle ranges.
    """
    yaw_min, yaw_max = -np.pi, np.pi
    start_angle = np.random.uniform(yaw_min, yaw_max)
    step = (yaw_max - yaw_min) / n

    ranges = []
    for i in range(n):
        range_start = start_angle + i * step
        range_end = start_angle + (i + 1) * step
        
        range_start = (range_start + np.pi) % (2 * np.pi) - np.pi
        range_end = (range_end + np.pi) % (2 * np.pi) - np.pi

        ranges.append((range_start, range_end))

    return ranges


def divide_point_cloud_pitch(n, pitch_range):
    """Divide the pitch angle range [pitch_min, pitch_max] into N equal parts evenly."""
    pitch_min, pitch_max = pitch_range
    start_angle = pitch_min
    step = (pitch_max - pitch_min) / n

    ranges = []
    for i in range(n):
        range_start = start_angle + i * step
        range_end = start_angle + (i + 1) * step
        ranges.append((range_start, range_end))

    return ranges   

def split_yaw(yaw, start, end):
    """
    Split the point cloud according to yaw angles by given range.
    """
    assert yaw.min() >= -np.pi and yaw.max() <= np.pi
    if start > end:
        return (yaw > start) | (yaw < end)
    else:
        return (yaw > start) & (yaw < end)
  
def find_closest_img(img_list, bbox):
    """
    Find the image from the img_list that is closest in size to the given bbox.

    Args:
        img_list (list): A list of images (each as a numpy array or similar format).
        bbox (list): The bounding box defined by 4 corner points, e.g., 
                     [[179, 485], [1599, 485], [1599, 678], [179, 678]].

    Returns:
        tuple: The closest image and its dimensions (width, height).
    """
    bbox_width = abs(bbox[1][0] - bbox[0][0])
    bbox_height = abs(bbox[2][1] - bbox[1][1])
    bbox_size = np.array([bbox_width, bbox_height])

    closest_img = None
    min_difference = float('inf')
    for img in img_list:
        img_height, img_width = img.shape[:2]
        img_size = np.array([img_width, img_height])
        size_difference = np.linalg.norm(bbox_size - img_size)

        if size_difference < min_difference:
            min_difference = size_difference
            closest_img = img
    return closest_img, (closest_img.shape[1], closest_img.shape[0])


def update_inst_img(input_dict, mix_results, coor_3d, instlabel, semlabel, 
                    source='newscan', id2img=None, draw_instlabel=None):
    assert source in ['newscan', 'presaved'], 'source should be either newscan or presaved.'
    assert coor_3d.shape[0] == instlabel.shape[0]
    assert coor_3d.shape[0] == semlabel.shape[0]
    if source == 'newscan':
        id2img = {} 
    
    out_im_list = input_dict['imgs']
    image_list_orig, image_list_mix = out_im_list, mix_results['imgs']
    image_list_orig = [torch.tensor(img) for img in image_list_orig] if isinstance(image_list_orig[0], np.ndarray) else image_list_orig
    image_list_mix  = [torch.tensor(img) for img in image_list_mix] if isinstance(image_list_mix[0], np.ndarray) else image_list_mix
    img_size = input_dict['img_shape'][:2] 
    ori_size = input_dict['ori_shape'] 
    img_scale = input_dict['scale_factor'][0] 
    assert int(img_size[0]) == int(ori_size[0] * img_scale)
    lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
    for i in range(len(image_list_orig)):
        im_orig, im_mix = image_list_orig[i].numpy(), image_list_mix[i].numpy()
        im_new = im_orig.copy()
        coor_2d, mask = proj_lidar2img(coor_3d, lidar2img_mix[i], 
                                            img_size=(ori_size[1], ori_size[0]), 
                                            min_dist=1.0)
        coor_2d = (coor_2d * img_scale).numpy()
        instlabel_v = instlabel[mask]
        semlabel_v  = semlabel[mask]
        if source == 'presaved':
            draw_instlabel_v = draw_instlabel[mask]

        for inst_id in np.unique(instlabel_v):
            inst_coor_2d = coor_2d[instlabel_v == inst_id]
            inst_cls = semlabel_v[instlabel_v == inst_id][0]
            box = fit_box_cv(inst_coor_2d)
            box_ex = expand_box(box, 20, im_mix.shape[:2]) 
            if source == 'newscan':
                crop_img = crop_box_img(box_ex, im_mix)
                if inst_id not in id2img:
                    id2img[inst_id] = [crop_img]
                else:
                    id2img[inst_id].append(crop_img)
            else:
                draw_id = draw_instlabel_v[instlabel_v == inst_id]
                assert np.unique(draw_id).shape[0] == 1, 'draw_id should be unique'
                draw_id = draw_id[0]
                if int(draw_id) not in id2img.keys():
                    print(f'Warning: draw_id {draw_id} not in id2img.keys(): {id2img.keys()}')
                    continue
                crop_img, _ = find_closest_img(id2img[int(draw_id)], box_ex)
                im_new = paste_box_img(box_ex, im_new, crop_img)
        if source == 'presaved':
            out_im_list[i] = im_new
    return out_im_list, id2img

