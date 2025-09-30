import io
import os
import torch
import numpy as np
from typing import Tuple
import requests
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops.boxes import batched_nms

from huggingface_hub import hf_hub_download
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import GroundingDINO.groundingdino.datasets.transforms as T

def lift_instance_mask(instance_mask, include_background=False):
    unique_ids = np.unique(instance_mask)
    if not include_background:
        unique_ids = unique_ids[unique_ids != 0]

    binary_masks = (instance_mask[None, :, :] == unique_ids[:, None, None]).astype(np.uint8)
    return binary_masks, unique_ids

def load_image_from_np(image:np.array) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = image[:, :, [2, 1, 0]] # BGR to RGB
    image_source = Image.fromarray(image)
    # image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    """Function `load_model_hf`.

Args:
    repo_id: 
    filename: 
    ckpt_config_filename: 
    device: """
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

def plot_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    # ax.imshow(img)
    # add the image to the plot
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
    
def show_all_mask(masks, scores, boxes, img):
    """show every mask on the image one by one"""
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        _show_mask(mask, plt.gca())
        # _show_box(boxes[i], plt.gca())
        # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()  
  
def _show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    if mask.device.type == 'cuda':
        mask = mask.cpu()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def _show_box(box, ax):

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def pack_anno(masks, scores, target_bboxes, obj_labels):
    anno = []
    for i, (mask, score, box, label) in enumerate(zip(masks, scores, target_bboxes, obj_labels)):
        anno.append({
            'segmentation': mask.cpu().numpy(),
            'scores': score.item(),
            'boxes': box.cpu().numpy(),
            'area': mask.sum().item(),
            'label': label
        })
    return anno

def keep_overlaped(boxes1, boxes2, logits1, logits2, phrases1, phrases2, threshold):
    assert boxes1.shape[1] == 4
    assert boxes2.shape[1] == 4
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.tensor([]), torch.tensor([]), []
    
    results_box, results_logit, results_phrase = [], [], []

    for i in range(boxes1.size(0)):
        x1, y1, w1, h1 = boxes1[i]
        x2, y2 = x1 + w1, y1 + h1
        
        x1_prime = boxes2[:, 0]
        y1_prime = boxes2[:, 1]
        w2_prime = boxes2[:, 2]
        h2_prime = boxes2[:, 3]
        x2_prime = x1_prime + w2_prime
        y2_prime = y1_prime + h2_prime
        
        inter_x1 = torch.max(x1, x1_prime)
        inter_y1 = torch.max(y1, y1_prime)
        inter_x2 = torch.min(x2, x2_prime)
        inter_y2 = torch.min(y2, y2_prime)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        
        inter_area = inter_w * inter_h
        area1 = w1 * h1
        area2 = w2_prime * h2_prime
        
        iou = inter_area / (area1 + area2 - inter_area)
        
        valid_mask = iou > threshold
        
        if valid_mask.any():
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze()
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            for idx in valid_indices:
                if logits1[i] >= logits2[idx]:
                    selected_box = boxes1[i]
                    selected_logit = logits1[i]
                    selected_phrase = phrases1[i]
                else:
                    selected_box = boxes2[idx]
                    selected_logit = logits2[idx]
                    selected_phrase = phrases2[idx]
                
                results_box.append(selected_box.unsqueeze(0))
                results_logit.append(selected_logit)
                results_phrase.append(selected_phrase)
    if len(results_box) == 0:
        return torch.tensor([]), torch.tensor([]), []
    results_box = torch.cat(results_box, dim=0)
    results_logit = torch.tensor(results_logit)
    return results_box, results_logit, results_phrase

def filter_boxes(boxes, logits, phrases, thing_texts, thre=0.7):
    '''filter out boxes that are in thing_texts by nms'''
    assert boxes.shape[0] == logits.shape[0] == len(phrases)

    if isinstance(phrases[0], list):
        print(f"Warning: phrases is nested list: {phrases}")
        phrases = [item[0] if isinstance(item, list) and len(item) > 0 else str(item) for item in phrases]
        print(f"Flattened phrases: {phrases}")

    phrases = [str(phrase) for phrase in phrases]

    phrase = phrases[0].replace(' - ', '-') # avoid 'xxx - x' occur
    if ' ' in phrase:
        phrase = phrase.replace(' ', '-') # avoid 'xxx x' occur
    if phrase=='safty-cone' or phrase=='traffic' or phrase=='cone':
        phrase = 'traffic-cone'
    try:
        d = thing_texts.index(phrase)
    except:
        d = 3 # error occurs when the phrase is 'traffic-cone'
    phrases_id = torch.ones_like(boxes[:, 0])*d
    
    keep_by_nms = batched_nms(
                boxes,
                logits,
                phrases_id,
                iou_threshold=thre,
            )
    # print(keep_by_nms)
    phrases_ = []
    for i in range(phrases_id[keep_by_nms].shape[0]):
        phrases_.append(thing_texts[int(phrases_id[i])])
    
    assert boxes[keep_by_nms].shape[0] == logits[keep_by_nms].shape[0] == len(phrases_)
    return boxes[keep_by_nms], logits[keep_by_nms], phrases_, keep_by_nms

def plot_point_in_camview(points, coloring, im, dot_size=5):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.ioff()

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # fig.canvas.set_window_title('Point cloud in camera view')

    ax.imshow(im)

    
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)

    ax.axis('off')

    fig.canvas.draw()

    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    plt.close(fig)

    return img_array

class PointCloudVisualizer:
    """Class `PointCloudVisualizer`."""
    def __init__(self, out_dict, color_map, save_path='output/rerun_samples/', dataset_type='semantickitti'):
        """Function `__init__`.

Args:
    out_dict: 
    color_map: 
    save_path: 
    dataset_type: """
        self.points = out_dict['points'].copy()
        if 'pts_semantic_mask' in out_dict:
            self.sem_mask = out_dict['pts_semantic_mask'].copy()
        if 'augment_mask' in out_dict:
            self.aug_mask = out_dict['augment_mask'].copy()
        if 'pts_instance_mask' in out_dict:
            self.inst_mask = out_dict['pts_instance_mask'].copy()
        if 'pred_inst' in out_dict:
            self.pred_inst = out_dict['pred_inst'].copy()
        self.color_map = color_map
        self.save_path = save_path
        if dataset_type == 'semantickitti':
            self.ignore_class = 19
            self.thing_classes = range(8)
        elif dataset_type == 'nuscenes':
            self.ignore_class = 0
            self.thing_classes = range(1, 11)

    def prepare_semantic_colors(self):
        """Function `prepare_semantic_colors`."""
        return np.array([self.color_map[i] for i in self.sem_mask]) / 255
    
    def prepare_semantic_thing_colors(self):
        """Function `prepare_semantic_thing_colors`."""
        sem_mask_thing = self.sem_mask.copy()
        # nusc
        # sem_mask_thing[(sem_mask_thing==0) | (sem_mask_thing>=10)] = 0
        # skitti
        # print(f'NOTE: Skitti semantic label is different from nuscenes. Please check the label mapping.')
        sem_mask_thing[~np.isin(sem_mask_thing, self.thing_classes)] = self.ignore_class
        return np.array([self.color_map[i] for i in sem_mask_thing]) / 255

    def prepare_augment_colors(self):
        """Function `prepare_augment_colors`."""
        aug_color = np.array([[255, 0, 0], [0, 0, 255]]) # 0: red, 1: blue
        aug_mask = self.aug_mask.astype(np.int8)
        return aug_color[aug_mask] / 255

    def prepare_instance_colors(self, inst_mask, filter_stuffs=False):
        """Function `prepare_instance_colors`.

Args:
    inst_mask: 
    filter_stuffs: """
        coloring_inst = np.ones((inst_mask.shape[0], 3))
        # colorize the background to light gray
        coloring_inst = coloring_inst* np.array([200, 200, 200]) / 255
        num_inst = 0
        for inst_id in np.unique(inst_mask):
            if inst_id == 0 or (filter_stuffs and np.unique(self.sem_mask[inst_mask == inst_id])[0] not in self.thing_classes):
                # print(f'Ignore instance {inst_id}, semantic label: {np.unique(self.sem_mask[inst_mask == inst_id])[0]} due to {inst_id == 0} or {np.unique(self.sem_mask[inst_mask == inst_id])[0] not in self.thing_classes')
                continue
            coloring_inst[inst_mask == inst_id] = np.random.randint(0, 255, 3) / 255
            num_inst += 1
        return coloring_inst, num_inst

    def save_data(self):
        """Function `save_data`."""
        np.save(self.save_path + 'points.npy', self.points)
        np.save(self.save_path + 'sem_mask.npy', self.sem_mask)
        np.save(self.save_path + 'aug_mask.npy', self.aug_mask)

    def plot(self, color_mode='instance', mode='2d', custom_input=None, filter_stuffs=False):
        """Function `plot`.

Args:
    color_mode: 
    mode: 
    custom_input: 
    filter_stuffs: """
        assert mode in ['2d', '3d'], "Invalid mode. Choose from '2d' or '3d'."
        if color_mode == 'semantic':
            colors = self.prepare_semantic_colors()
        elif color_mode == 'semantic_thing':
            colors = self.prepare_semantic_thing_colors()
        elif color_mode == 'augment':
            colors = self.prepare_augment_colors()
        elif color_mode == 'instance':
            colors, num_inst = self.prepare_instance_colors(self.inst_mask, filter_stuffs)
            print(f'Number of instances: {num_inst}')
        elif color_mode == 'pred_instance':
            colors, num_inst = self.prepare_instance_colors(self.pred_inst, filter_stuffs)
            print(f'Number of instances: {num_inst}')
        elif color_mode == 'custom':
            colors, num_inst = self.prepare_instance_colors(custom_input, filter_stuffs)
            print(f'Number of instances: {num_inst}')
        else:
            raise ValueError("Invalid color_mode. Choose from 'semantic', 'augment', or 'instance'.")

        fig = plt.figure(figsize=(10, 10))
        if mode == '2d':
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.scatter(self.points[:, 0], self.points[:, 1], c=colors, s=1)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
        else:
            ax = fig.add_subplot(111, projection=mode, proj_type='ortho')
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=colors, s=3)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        plt.show()

def save_masks(masks, class_scores, mask_scores, bboxes, obj_labels,
                                          save_path, thing_texts, scale_factor=0.4):
    save_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "annotation"), exist_ok=True)

    masks_np = masks.cpu().numpy()
    n, h, w = masks_np.shape

    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    instance_mask = np.zeros((new_h, new_w), dtype=np.uint16)

    annotations = []

    for i in range(n):
        resized_mask = resize(masks_np[i], (new_h, new_w),
                            order=0, preserve_range=True, anti_aliasing=False).astype(bool)

        instance_mask[resized_mask] = i + 1

        try:
            category_id = thing_texts.index(obj_labels[i])
        except ValueError:
            category_id = len(thing_texts)

        annotation = {
            "id": i + 1,
            "category_id": category_id,
            "class_name": obj_labels[i],
            "class_score": float(class_scores[i].cpu()),
            "mask_score": float(mask_scores[i].cpu().squeeze()),
            "bbox": [float(x) for x in bboxes[i].cpu().tolist()],  # [x1, y1, x2, y2]
            "area": int(resized_mask.sum())
        }
        annotations.append(annotation)

    mask_save_path = os.path.join(save_dir, "mask", f"{base_name}.png")
    Image.fromarray(instance_mask).save(mask_save_path)

    json_save_path = os.path.join(save_dir, "annotation", f"{base_name}.json")

    coco_data = {
        "info": {
            "description": "3D Multi-Modal Panoptic Segmentation Results",
            "version": "1.0",
            "year": 2024
        },
        "images": [
            {
                "id": 1,
                "width": new_w,
                "height": new_h,
                "file_name": f"{base_name}_masks.png"
            }
        ],
        "categories": [
            {"id": i, "name": name} for i, name in enumerate(thing_texts)
        ],
        "annotations": annotations
    }

    with open(json_save_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

def load_masks(save_path):
    import json
    import numpy as np
    from PIL import Image
    import os
    import torch

    save_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    mask_path = os.path.join(save_dir, 'mask', f"{base_name}.png")
    json_path = os.path.join(save_dir, 'annotation', f"{base_name}.json")

    assert os.path.exists(mask_path), f"Mask file not found: {mask_path}"
    assert os.path.exists(json_path), f"JSON file not found: {json_path}"

    mask_img = Image.open(mask_path)
    instance_mask = np.array(mask_img)

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    annotations = coco_data['annotations']
    n_instances = len(annotations)

    if n_instances == 0:
        h, w = instance_mask.shape
        return (torch.zeros((0, h, w), dtype=torch.bool),
                torch.zeros((0,)),
                torch.zeros((0, 1)),
                torch.zeros((0, 4)),
                [],
                coco_data)

    h, w = instance_mask.shape

    masks = torch.zeros((n_instances, h, w), dtype=torch.bool)
    class_scores = torch.zeros(n_instances)
    mask_scores = torch.zeros((n_instances, 1))
    bboxes = torch.zeros((n_instances, 4))
    obj_labels = []

    for i, ann in enumerate(annotations):
        instance_id = ann['id']

        mask = (instance_mask == instance_id)
        masks[i] = torch.from_numpy(mask)

        class_scores[i] = ann['class_score']
        mask_scores[i, 0] = ann['mask_score']
        bboxes[i] = torch.tensor(ann['bbox'])
        obj_labels.append(ann['class_name'])

    return masks, class_scores, mask_scores, bboxes, obj_labels, coco_data

def save_plot(left_image, right_image, left_title, right_title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 9))

    ax1.imshow(left_image)
    ax1.set_title(left_title, fontsize=14)
    ax1.axis('off')

    ax2.imshow(right_image)
    ax2.set_title(right_title, fontsize=14)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def filter_overlap(masks, class_scores, obj_labels, overlap_threshold=0.5):

    n = masks.shape[0]
    if n <= 1:
        return masks, class_scores, obj_labels, torch.arange(n)

    masks_flat = masks.view(n, -1).float()  # (N, H*W)

    intersections = torch.mm(masks_flat, masks_flat.t())  # (N, N)
    areas = masks_flat.sum(dim=1)  # (N,)
    unions = areas.unsqueeze(0) + areas.unsqueeze(1) - intersections  # (N, N)

    ious = intersections / (unions + 1e-8)

    triu_mask = torch.triu(torch.ones_like(ious, dtype=torch.bool), diagonal=1)
    overlapping_pairs = torch.where((ious > overlap_threshold) & triu_mask)

    to_remove = set()

    for i, j in zip(overlapping_pairs[0], overlapping_pairs[1]):
        i, j = i.item(), j.item()

        if i not in to_remove and j not in to_remove:
            overlap_ratio = ious[i, j].item()
            score_i = class_scores[i].item()
            score_j = class_scores[j].item()
            if score_i >= score_j:
                to_remove.add(j)
            else:
                to_remove.add(i)

    keep_indices = torch.tensor([i for i in range(n) if i not in to_remove], dtype=torch.long)

    if len(keep_indices) == 0:
        best_idx = torch.argmax(class_scores)
        keep_indices = torch.tensor([best_idx])

    filtered_masks = masks[keep_indices]
    filtered_scores = class_scores[keep_indices]
    filtered_labels = [obj_labels[i] for i in keep_indices]

    return filtered_masks, filtered_scores, filtered_labels, keep_indices

def filter_points(points_3d, poi, sam_mask, view_mask, lidar_idx, sem_mask):
    poi = poi.round().to(torch.int64)
    n, h, w = sam_mask.shape
    N = points_3d.shape[0]
    sel_idx = torch.arange(0, N, device=poi.device)[view_mask]
    assert (poi[:, 0] >= 0).all() and (poi[:, 0] < h).all() and (poi[:, 1] >= 0).all() and (poi[:, 1] < w).all()
    
    visible_lidar_idx = lidar_idx[view_mask].to(torch.int64)
    visible_sem_mask = torch.tensor(sem_mask[view_mask]).to(poi.device).to(torch.int64)
    mask_values = sam_mask[:, poi[:, 0], poi[:, 1]]
    
    obj_coors = []
    obj_idxs = []
    obj_labels = []
    for i in range(n):
        mask_indices = mask_values[i] == 1
        obj_coors.append(poi[mask_indices])
        obj_idxs.append(sel_idx[mask_indices])
        obj_labels.append(visible_sem_mask[mask_indices])
        
    return obj_coors, obj_idxs, obj_labels
