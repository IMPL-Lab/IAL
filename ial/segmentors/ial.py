import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter_max

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import convert_sync_batchnorm
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet.models.utils import multi_apply
from tools.projection import proj_lidar2img
from .p3former import _P3Former

@MODELS.register_module()
class IAL(_P3Former):
    def __init__(self,
                 dataset_type: str,
                 data_preprocessor: OptConfigType,
                 backbone: ConfigType,
                 img_backbone: ConfigType,
                 feat_dims: list, 
                 decode_head: ConfigType,
                 voxel_encoder: ConfigType,
                 neck: OptConfigType = None,
                 embed_dims: int = 256,
                 enable_sync_bn: bool = False,
                 inst_3d_params: dict = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(IAL, self).__init__(
            data_preprocessor=data_preprocessor,
            backbone=backbone,
            decode_head=decode_head,
            voxel_encoder=voxel_encoder,
            neck=neck,
            loss_regularization=loss_regularization,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg
        )
        assert dataset_type in ['nuscenes', 'semantickitti'], 'dataset can only be nuscenes or semantickitti'
        self.dataset_type = dataset_type
        self.img_backbone = MODELS.build(img_backbone)
        self.feat_dims = feat_dims
        self.embed_dims = embed_dims
        self.mm_conv = nn.ModuleList()
        for feat_dim in self.feat_dims:
            self.mm_conv.append(nn.Sequential(
                nn.Linear(feat_dim, self.embed_dims, bias=False),
                build_norm_layer(dict(type='LN'), self.embed_dims)[1],
                build_activation_layer(dict(type='GELU'))))
        self.inst_3d_params = inst_3d_params
        self.enable_sync_bn = enable_sync_bn
        if self.enable_sync_bn:
            self.img_backbone = convert_sync_batchnorm(self.img_backbone, 'torch')

    def extract_feat(self, batch_inputs: dict, batch_data_samples) -> Tensor:
        encoded_feats = self.voxel_encoder(batch_inputs['voxels']['voxels'], batch_inputs['voxels']['coors'])
        batch_inputs['voxels']['voxel_coors'] = encoded_feats[1]
        batch_inputs['points'] = [points[:, :3] for points in batch_inputs['points']] 
        
        x = self.backbone(encoded_feats[0], encoded_feats[1], len(batch_inputs['points']), batch_data_samples)
        if self.with_neck:
            x = self.neck(x)
        voxel_features, heatmap, fog = x
        hm_outputs = {}
        hm_outputs['heatmap'] = heatmap
        hm_outputs['fog'] = fog
        pois, poi_masks = self.worldc_to_camc(batch_inputs['points'], batch_data_samples)
        
        imgs = batch_inputs['imgs']  
        B, N, C, H, W = imgs.size()
        img_feats = self.img_backbone(imgs.view(B * N, C, H, W))
        img_feats = img_feats[0].view(B, N, *img_feats[0].shape[1:])
        img_feats_list = [img_feats[bs] for bs in range(B)]
        
        pimg_feats, _ = multi_apply(self.extract_aligned_imgfeat, pois, poi_masks, batch_data_samples, img_feats_list)
        p2v_maps = [ds.gt_pts_seg.point2voxel_map for ds in batch_data_samples]
        vimg_feats, _ = multi_apply(self.p2v_aggregate, pimg_feats, p2v_maps)
        batch_inputs['img_features'] = vimg_feats
        
        return voxel_features, hm_outputs

    def loss(self, batch_inputs_dict, batch_data_samples):
        x, hm_outputs = self.extract_feat(batch_inputs_dict, batch_data_samples)
        batch_inputs_dict['features'] = x.features
        self.pack_heatmap_fog(hm_outputs, batch_inputs_dict)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)
        return losses
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        x, hm_outputs = self.extract_feat(batch_inputs_dict, batch_data_samples)
        batch_inputs_dict['features'] = x.features
        self.pack_heatmap_fog(hm_outputs, batch_inputs_dict)
        
        pts_semantic_preds, pts_instance_preds = self.decode_head.predict(batch_inputs_dict, batch_data_samples)
        return self.postprocess_result(pts_semantic_preds, pts_instance_preds, batch_data_samples)

    def worldc_to_camc(self, voxel_coors_list, batch_data_samples):
        num_view = 6 if self.dataset_type == 'nuscenes' else 2
        pois, batch_voxel_masks = [], []
        batch_size = len(batch_data_samples)
        for bs, voxel_coor, data_sample in zip(range(batch_size), voxel_coors_list, batch_data_samples):
            valid_coords, masks = [], []
            for v in range(num_view):
                lidar2img = data_sample.img_metas['lidars2imgs_torch'].to(voxel_coor.device)
                ori_size = data_sample.img_metas['ori_shape']
                voxel_cam_coord, mask = proj_lidar2img(voxel_coor.detach()[:,:3], lidar2img[v], 
                                                    img_size=(ori_size[1], ori_size[0]), 
                                                    min_dist=1.0)
                voxel_cam_coord[:, [0, 1]] = voxel_cam_coord[:, [1, 0]] 
                valid_coords.append(voxel_cam_coord)
                masks.append(mask)
            pois.append(valid_coords)
            batch_voxel_masks.append(masks)
            
        return pois, batch_voxel_masks
    
    def extract_aligned_imgfeat(self, pois, masks, data_sample, img_feats, num_layers=1):
        num_voxel = masks[0].shape[0]
        cam_feat = torch.zeros((num_voxel, self.embed_dims), device=img_feats[0].device).float() 
        mask_cnt = torch.zeros((num_voxel, 1)).to(cam_feat.device)

        for i, (valid_coord, mask) in enumerate(zip(pois, masks)): 
            up_size = data_sample.batch_input_shape
            scale = torch.tensor(data_sample.scale_factor[0]).to(valid_coord.device) 
            feat_idx = torch.floor(valid_coord * scale).long()
            
            img_feat_rz = F.interpolate(img_feats[i, :, :, :].unsqueeze(0), 
                                             size=up_size, mode='bilinear', 
                                             align_corners=False).squeeze(0) 
            cam_feat_per_voxel = img_feat_rz[:, feat_idx[:, 0], feat_idx[:, 1]]  
            cam_feat_per_voxel = cam_feat_per_voxel.T
            mask_cnt += mask.reshape(-1, 1)
            cam_feat_per_voxel = self.mm_conv[num_layers-1](cam_feat_per_voxel) 
            cam_feat[torch.where(mask>0)[0], :] += cam_feat_per_voxel
                
        mask_cnt[torch.where(mask_cnt==0)[0]] = 1 # avoid divide by zero
        cam_feat = (cam_feat/mask_cnt).float()             
        return cam_feat, mask_cnt

    def p2v_aggregate(self, point_feat, point2voxel_map):
        point2voxel_map = point2voxel_map.to(torch.int64).to(point_feat.device)
        num_voxel = point2voxel_map.max() + 1
        feat_dim = point_feat.shape[1]
        mapping_expanded = point2voxel_map.unsqueeze(1).expand(-1, feat_dim).T
        point_feat = point_feat.float().T  # (feat_dim, N_p)
        
        voxel_feat, _ = scatter_max(point_feat, mapping_expanded, 
                                  out=point_feat.new_zeros((feat_dim, num_voxel)))
        return voxel_feat.T, mapping_expanded.T  # (N_v, feat_dim)

    def pack_heatmap_fog(self, hm_outputs: dict, batch_inputs_dict: dict) -> None:
        for key in ['heatmap', 'fog']:
            batch_inputs_dict[f'pred_{key}'] = hm_outputs[key]