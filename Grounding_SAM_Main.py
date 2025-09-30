import os
import argparse
import sys
import numpy as np
import torch
import argparse
import pickle
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import h5py
import matplotlib.pyplot as plt

from groundingdino.util import box_ops
from groundingdino.util.inference import annotate, load_image, predict
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_warning()
from segment_anything.segment_anything import build_sam, SamPredictor 
from Grounding_SAM_Utils import load_model_hf, plot_anns, pack_anno, filter_boxes, plot_point_in_camview, save_masks, load_masks, filter_overlap, filter_points, save_plot

sys.path.append('tools')
from general_tool import load_point, load_mask, expand_mask_torch, load_viewimages_path
from projection import proj_lidar2img_nusc 
from query_generator import cand_recall_calc

# hyperparameters
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
overlap_thre = 0.8
dino_nms_thresh = 0.3
ENABLE_MASK_OVERLAP_FILTER = True  
MASK_OVERLAP_THRESHOLD = 0.8  
sam_folder = 'gsam'
pred_2d_inst_name = 'pred_inst_pts_xcluster'
SAVE_IMAGES = True
VIEWS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
all_texts = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'person', 'safty-cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
thing_texts = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'person', 'safty-cone', 'trailer', 'truck']
text_labels = {'barrier': 1, 'bicycle': 2, 'bus': 3, 'car': 4, 'construction_vehicle': 5, 'motorcycle': 6, 'person': 7, 'safty-cone': 8, 'trailer': 9, 'truck': 10}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system parameters
parser = argparse.ArgumentParser(description='Customized Grounded-Segment-Anything for 3DMM')
parser.add_argument('--data_mode', type=str, default='mini', help='trainval or mini')
parser.add_argument('--dset', type=str, default='train_1', help='train or val or train_1')
parser.add_argument('--part', type=str, default='1/1', help='split the dataset to parts')
dmode = parser.parse_args().data_mode
dset = parser.parse_args().dset
part = parser.parse_args().part

proj_path = os.path.join(os.getcwd()+'/') 
if dmode=='trainval':
    data_root = 'data/nuscenes_full' 
else: 
    data_root = 'data/nuscenes_mini'
nusc = NuScenes(version=f'v1.0-{dmode}', dataroot=proj_path+data_root, verbose=True)
pklfile = os.path.join(proj_path, f'{data_root}/nuscenes_infos_{dset}.pkl')
with open(pklfile, 'rb') as f:
    data = pickle.load(f)
res_list = load_viewimages_path(data, data_root)

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
sam_checkpoint_path = 'checkpoints/sam/sam_vit_h_4b8939.pth'
sam_checkpoint = sam_checkpoint_path
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

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

color_map = {
            0: [0, 0, 0],  # noise                 black
            1: [255, 120, 50],  # barrier               orange
            2: [255, 192, 203],  # bicycle               pink
            3: [255, 255, 0],  # bus                   yellow
            4: [0, 150, 245],  # car                   blue
            5: [0, 255, 255],  # construction_vehicle  cyan
            6: [255, 127, 0],  # motorcycle            dark orange
            7: [255, 0, 0],  # pedestrian            red
            8: [255, 240, 150],  # traffic-cone          light yellow
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

folder_path = data['data_list'][0]['images']['CAM_FRONT']['img_path'].split('samples')[0]+sam_folder+'/'
if not os.path.exists(folder_path): 
    os.mkdir(folder_path) 
for p in ['sam_ground_seg', 'sam_ground_det', 'sam_ground_pth']:
    if not os.path.exists(folder_path+p):
        os.mkdir(folder_path+p)
    for v in VIEWS:
        path = folder_path+p+'/'+v
        if not os.path.exists(path):
            print(f'Make dir: {path}')
            os.mkdir(path) 

tp, total, pred = 0, 0, 0
gt_p_b, pred_p_b, gt_p_s, pred_p_s = [], [], [], []
missed = []
if part == '1/1':
    start_idx = 0
    end_idx = len(data['data_list'])
else:
    ptarget = int(part.split('/')[0])-1
    ptotal = int(part.split('/')[1])
    num_total = len(data['data_list'])
    start_idx = int(ptarget)*(num_total//int(ptotal))
    end_idx = (int(ptarget)+1)*(num_total//int(ptotal)) 
    if int(ptarget) == int(ptotal)-1:
        end_idx = num_total
        print(f'Last part: {start_idx} to {end_idx}')
print('##############################################')
print(f'Processing from {start_idx} to {end_idx}.')
print('##############################################')
for i in tqdm(range(start_idx, end_idx)):
    save_path = res_list[i]['lidar_path'].replace('samples/LIDAR_TOP', f'{sam_folder}/{pred_2d_inst_name}').replace('.pcd.bin', '.h5') 
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if os.path.exists(save_path):
        continue
    
    if 'lidar_path' in res_list[i].keys():
        lidar_point = load_point(res_list[i]['lidar_path'])
    else:
        lidar_path = res_list[i]['lidar_path']
        lidar_point = load_point(lidar_path)
    dataset_path = data_prefix.get('dataset_path')
    panoptic_path = data['data_list'][i]['pts_panoptic_mask_path']
    panoptic_path = os.path.join(dataset_path, panoptic_path)
    pts_semantic_mask, pts_inst_mask = load_mask(panoptic_path)
    pts_semantic_mask = np.vectorize(label_map.get)(pts_semantic_mask)
    sem_mask = pts_semantic_mask
    sample_token_v = data['data_list'][i]['token']
    point_coor = np.concatenate((lidar_point, np.ones((lidar_point.shape[0], 1))), axis=1)

    N = lidar_point.shape[0]
    lidar_idx = torch.arange(0, N).to(DEVICE)# view = 'CAM_FRONT_LEFT'
    lidar_cluster = {}
    lidar_cluster_labels = {}
    lidar_cluster_class_scores = {}
    lidar_cluster_mask_scores = {}
    lidar_cluster_classes = {}
    cluster_num = 1
    
    for v in VIEWS: 
        save_seg_path = os.path.join(proj_path, data['data_list'][i]['images'][v]['img_path'].replace('samples', sam_folder+'/sam_ground_seg'))
        save_det_path = os.path.join(proj_path, data['data_list'][i]['images'][v]['img_path'].replace('samples', sam_folder+'/sam_ground_det'))
        save_pth_path = os.path.join(proj_path, data['data_list'][i]['images'][v]['img_path'].replace('samples', sam_folder+'/2d_results').replace('.jpg', '.pth'))
        
        sample_v = nusc.get('sample', sample_token_v)
        pointsensor_token_v = nusc.get('sample_data', sample_v['data']['LIDAR_TOP'])
        point_calibrated_sensor_token_v, point_ego_pose_token_v = pointsensor_token_v['calibrated_sensor_token'], pointsensor_token_v['ego_pose_token']
        cam_token_v = sample_v['data'][v]
        point_cam_coords1, coloring_p, im_p, mask_p = proj_lidar2img_nusc(
            nusc, point_coor, 
            point_calibrated_sensor_token_v, point_ego_pose_token_v, 
            cam_token_v, min_dist=1.0, return_img=True,
            sem_mask=sem_mask,
            use_label_map=False)
        
        thing_mask = (sem_mask>0)&(sem_mask<11)
        coloring_p[~thing_mask] = [0, 0, 0] # ignore stuffs
        coor = torch.tensor(point_coor[mask_p]).to(DEVICE)
        poi = point_cam_coords1[:, mask_p][:2, :].T.astype(int) # (N, 2)
        poi = torch.tensor(poi).to(DEVICE)
        poi = poi[:,[1,0]]
        
        img_path = res_list[i]['imgs_meta']['img_path'][v]
        image_source, image = load_image(img_path)
        H, W, _ = image_source.shape
        target_bboxes, obj_labels, all_cls_scores = [], [], []
        annotated_frame = image_source.copy()
        
        save_dir = os.path.dirname(save_pth_path)
        base_name = os.path.splitext(os.path.basename(save_pth_path))[0]
        mask_path = os.path.join(save_dir, 'mask', f"{base_name}.png")
        json_path = os.path.join(save_dir, 'annotation', f"{base_name}.json")

        if os.path.exists(mask_path) and os.path.exists(json_path):
            sam_masks, det_scores, mask_scores, target_bboxes, classes, coco_data = load_masks(save_pth_path)
            sam_masks = sam_masks.to(DEVICE)
            det_scores = det_scores.to(DEVICE) if isinstance(det_scores, torch.Tensor) else torch.tensor(det_scores).to(DEVICE)
            mask_scores = mask_scores.to(DEVICE) if isinstance(mask_scores, torch.Tensor) else torch.tensor(mask_scores).to(DEVICE)
            sam_masks = expand_mask_torch(sam_masks, scale_factor=1/0.4, dim_expand=False)

        else:
            for d, TEXT_PROMPT in enumerate(thing_texts):                
                
                boxes, logits, phrases = predict(
                    model=groundingdino_model, 
                    image=image, 
                    caption=TEXT_PROMPT, 
                    box_threshold=BOX_TRESHOLD, 
                    text_threshold=TEXT_TRESHOLD,
                    device=DEVICE
                )
                if boxes.shape[0] == 0:
                    continue

                H, W, _ = image_source.shape
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                boxes_xyxy, cls_scores, phrases, keep = filter_boxes(boxes_xyxy, logits, phrases, thing_texts=thing_texts, thre=dino_nms_thresh)
                boxes = boxes[keep]
                assert len(boxes) == len(cls_scores) == len(phrases)

                if len(boxes) == 0:
                    continue
                
                if SAVE_IMAGES:
                    annotated_frame = annotate(image_source=annotated_frame, boxes=boxes, logits=cls_scores, phrases=phrases)
                    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
                target_bboxes.append(boxes_xyxy)
                all_cls_scores.append(cls_scores)
                for _ in range(boxes_xyxy.shape[0]):
                    obj_labels.append(TEXT_PROMPT)
                
            if SAVE_IMAGES: 
                plt.figure(figsize=(16, 9))
                plt.imshow(annotated_frame)
                plt.axis('off')
                plt.savefig(save_det_path)
                plt.close()

                if target_bboxes != []:
                    target_bboxes = torch.cat(target_bboxes, dim=0)
                    all_cls_scores = torch.cat(all_cls_scores, dim=0)
                else:
                    res = np.array(image_source)
                    img_rendered_poi = plot_point_in_camview(point_cam_coords1[:, mask_p], coloring_p[mask_p, :], im_p, dot_size=4)

                    save_plot(
                        res, img_rendered_poi,
                        'Image (No Detections)', 'Point Cloud Projection on RGB Image',
                        save_seg_path
                    )
                    continue
            else:
                if target_bboxes != []:
                    target_bboxes = torch.cat(target_bboxes, dim=0)
                    all_cls_scores = torch.cat(all_cls_scores, dim=0)

            sam_predictor.set_image(image_source)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(target_bboxes, image_source.shape[:2]).to(DEVICE)
            masks, mask_scores, mask_logits = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        multimask_output = False,
                    )

            masks_ = masks.squeeze(1)
            if SAVE_IMAGES:
                res_with_masks = plot_anns(pack_anno(masks_, mask_scores, target_bboxes, obj_labels), image_source)

                img_rendered_poi = plot_point_in_camview(point_cam_coords1[:, mask_p], coloring_p[mask_p, :], im_p, dot_size=4)

                save_plot(
                    res_with_masks, img_rendered_poi,
                    'Image with Mask Annotations', 'Point Cloud Projection on RGB Image',
                    save_seg_path
                )
                
            masks_ = masks_.cpu()
            det_scores = all_cls_scores.cpu()
            mask_scores = mask_scores.cpu()
            mask_logits = mask_logits.cpu()
            target_bboxes = target_bboxes.cpu()

            if ENABLE_MASK_OVERLAP_FILTER and masks_.shape[0] > 1:
                filtered_masks, filtered_det_scores, filtered_obj_labels, keep_indices = filter_overlap(
                    masks_, det_scores, obj_labels, overlap_threshold=MASK_OVERLAP_THRESHOLD
                )

                filtered_mask_scores = mask_scores[keep_indices]
                filtered_mask_logits = mask_logits[keep_indices]
                filtered_target_bboxes = target_bboxes[keep_indices]

                masks_ = filtered_masks
                det_scores = filtered_det_scores
                mask_scores = filtered_mask_scores
                mask_logits = filtered_mask_logits
                target_bboxes = filtered_target_bboxes
                obj_labels = filtered_obj_labels
            
            debug_save_path = save_pth_path.replace('.pth', '_mask_debug.png')

            save_masks(
                masks_, det_scores, mask_scores, target_bboxes, obj_labels,
                save_pth_path, thing_texts, scale_factor=0.4
            )
            sam_masks = masks_.squeeze(1).to(DEVICE)
            classes = obj_labels

        obj_poi_coors, obj_poi_idx, obj_poi_labels = filter_points(
            point_coor, poi, sam_masks, mask_p, lidar_idx, sem_mask
        )
        
        for k in range(len(obj_poi_coors)):
            if obj_poi_coors[k].shape[0] == 0:
                continue
            lidar_cluster[cluster_num] = obj_poi_idx[k]  
            lidar_cluster_labels[cluster_num] = obj_poi_labels[k] 
            lidar_cluster_class_scores[cluster_num] = det_scores[k]
            lidar_cluster_mask_scores[cluster_num] =  mask_scores[k]
            lidar_cluster_classes[cluster_num] = classes[k]
            cluster_num += 1
                
    pts_inst_cluster = torch.zeros((N)).to(DEVICE)
    pts_inst_cluster_color = torch.zeros((N,3)).to(DEVICE)
    pts_inst_cluster_class_score = torch.zeros((N)).to(DEVICE)
    pts_inst_cluster_mask_score = torch.zeros((N)).to(DEVICE)
    pts_inst_cluster_classes = torch.zeros((N)).to(DEVICE)
    for n in lidar_cluster.keys():
        semantic_label = lidar_cluster_labels[n]
        pts_inst_cluster[lidar_cluster[n].squeeze()] = n
        pts_inst_cluster_class_score[lidar_cluster[n].squeeze()] = lidar_cluster_class_scores[n]
        pts_inst_cluster_mask_score[lidar_cluster[n].squeeze()] = lidar_cluster_mask_scores[n]
        pts_inst_cluster_classes[lidar_cluster[n].squeeze()] = text_labels[lidar_cluster_classes[n]]
        color = torch.zeros((semantic_label.shape[0], 3)).to(DEVICE)
        for idx, p in enumerate(semantic_label):
            color[idx] = torch.tensor(color_map[int(p)])/255
        pts_inst_cluster_color[lidar_cluster[n].squeeze()] = color
            
    # save the cluster results
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('mask', data=pts_inst_cluster.cpu().numpy().astype(np.uint16), compression='gzip')
        f.create_dataset('class_score', data=pts_inst_cluster_class_score.cpu().numpy().astype(np.float32), compression='gzip')
        f.create_dataset('mask_score', data=pts_inst_cluster_mask_score.cpu().numpy().astype(np.float32), compression='gzip')
        f.create_dataset('classes', data=pts_inst_cluster_classes.cpu().numpy().astype(np.uint8), compression='gzip')

    # calculate precision and recall
    ## load gt
    if 'pts_instance_mask' in res_list[i].keys():
        pts_instance_mask = res_list[i]['pts_instance_mask']
        pts_semantic_mask = res_list[i]['pts_semantic_mask']
    else:
        pts_semantic_mask, pts_instance_mask = load_mask(panoptic_path)
        pts_semantic_mask = np.vectorize(label_map.get)(pts_semantic_mask) # map to 0-19 classes
    
    gt_pts_inst_thing = pts_instance_mask.copy()
    gt_pts_inst_thing[(pts_semantic_mask==0)|(pts_semantic_mask>10)] = 0
    gt_pts_inst_thing = torch.from_numpy(gt_pts_inst_thing).to(DEVICE)
    
    ## load pred
    pred_pts_inst = torch.from_numpy(pts_inst_cluster).to(DEVICE) if isinstance(pts_inst_cluster, np.ndarray) else pts_inst_cluster.to(DEVICE)
    recall = cand_recall_calc(gt_pts_inst_thing, pred_pts_inst)
    tp += recall.sum().float()
    pred += len(torch.unique(pred_pts_inst))-1
    total += recall.shape[0]

print('$'*20)
print('precision dscore:', tp/pred, tp.cpu().numpy(), '/', pred)
print('recall dscore:', tp/total, tp.cpu().numpy(), '/', total)
