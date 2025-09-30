import torch
import torch.nn as nn
import numpy as np
from mmdet3d.registry import MODELS
from mmdet.models.utils import multi_apply

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops import SubMConv3d

from .p3former_head import _P3FormerHead, MLP
from tools.query_generator import extract_selected_features, select_2DQ, select_3DQ, fps_sample
from tools.voxelization import get_center_scale_voxel

# for multi-modal 
@MODELS.register_module()
class IALHead(_P3FormerHead):
    """P3Former head for 3D panoptic segmentation."""

    def __init__(self,
                num_classes,
                num_queries,
                embed_dims,
                thing_class,
                stuff_class,
                ignore_index,
                inst_3d_params=None,
                num_decoder_layers=3,
                pos_dim=3,
                transformer_decoder_cfg=dict(type='_Transformer_Decoder'),
                loss_cls=None,
                loss_mask=None,
                loss_dice=None,
                use_sem_loss=True,
                assigner_zero_layer_cfg=None,
                assigner_cfg=None,
                sampler_cfg=None,
                cls_channels=(128, 128, 20),
                mask_channels=(128, 128, 128, 128, 128),
                pe_type='mpe',
                use_pa_seg=True,
                pa_seg_weight = 0.2,
                score_thr=0.4,
                iou_thr=0.8,
                mask_score_thr=0.5,
                grid_size=[480, 360, 32],
                point_cloud_range=[],
                point_cloud_range_cart=[],):
        super(IALHead, self).__init__(
            num_classes=num_classes,
            num_queries=num_queries,
            embed_dims=embed_dims,
            thing_class=thing_class,
            stuff_class=stuff_class,
            ignore_index=ignore_index,
            num_decoder_layers=num_decoder_layers,
            pos_dim=pos_dim,
            transformer_decoder_cfg=transformer_decoder_cfg,
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            use_sem_loss=use_sem_loss,
            assigner_zero_layer_cfg=assigner_zero_layer_cfg,
            assigner_cfg=assigner_cfg,
            sampler_cfg=sampler_cfg,
            cls_channels=cls_channels,
            mask_channels=mask_channels,
            pe_type=pe_type,
            use_pa_seg=use_pa_seg,
            pa_seg_weight=pa_seg_weight,
            score_thr=score_thr,
            iou_thr=iou_thr,
            mask_score_thr=mask_score_thr,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range
        )
        
        self.inst_3d_params = inst_3d_params
        self.num_queries = num_queries
        self.thing_class = thing_class
        self.embed_dims = embed_dims
        self.point_cloud_range_polar = point_cloud_range
        self.point_cloud_range_cart = point_cloud_range_cart
        
        self.mm_conv = None 
        self.feat_fuse_layer = nn.Sequential(nn.Linear(embed_dims*2, embed_dims, bias=False))
        self.loss_heatmap = MODELS.build(inst_3d_params['loss_heatmap'])
        self.loss_fog = MODELS.build(inst_3d_params['loss_fog'])
                
        # query setttings
        self.queries = SubMConv3d(embed_dims, num_queries, indice_key="logit", # only for random queries
                                        kernel_size=1, stride=1, padding=0, bias=False)
            
        # query position encoding
        self.polar_proj_q = nn.Linear(pos_dim, embed_dims)
        self.polar_norm_q = build_norm_layer(dict(type='LN'), embed_dims)[1]
        self.cart_proj_q = nn.Linear(pos_dim, embed_dims)
        self.cart_norm_q = build_norm_layer(dict(type='LN'), embed_dims)[1]

        self.fc_coor_scale_lidar = MLP((1, embed_dims, embed_dims))
        self.polar_proj = nn.Linear(pos_dim, embed_dims)
        self.polar_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]
        self.cart_proj = nn.Linear(pos_dim, embed_dims)
        self.cart_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]

        self.pe_conv = nn.ModuleList()
        self.pe_conv.append(
            nn.Linear(embed_dims, embed_dims, bias=False))
        self.pe_conv.append(
            build_norm_layer(dict(type='LN'), embed_dims)[1])
        self.pe_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),))    
        self.position_proj = None
        self.position_norm = None
        self.feat_conv = None
        self.position_proj_im = self.position_proj
        self.position_norm_im = self.position_norm
        self.feat_conv_im = self.feat_conv
        self.polar_proj_im = self.polar_proj
        self.polar_norm_im = self.polar_norm
        self.cart_proj_im = self.cart_proj
        self.cart_norm_im = self.cart_norm
        self.pe_conv_im = self.pe_conv
        
        # token PE
        self.pe_conv_q = nn.ModuleList()
        self.pe_conv_q.append(
            nn.Linear(embed_dims, embed_dims, bias=False))
        self.pe_conv_q.append(
            build_norm_layer(dict(type='LN'), embed_dims)[1])
        self.pe_conv_q.append(build_activation_layer(dict(type='ReLU', inplace=True),))    
        self.position_proj_q = None
        self.position_norm_q = None
        self.feat_conv_q = None
        self.fc_coor_scale_q = MLP((1, embed_dims, embed_dims))   # [1, 256, 256]  

        self.feat_fuse_layer = nn.Sequential(
            nn.Linear(embed_dims*2, embed_dims*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims*4, embed_dims, bias=False),)
        self.pe_fuse_layer = nn.Sequential(
            nn.Linear(embed_dims*2, embed_dims*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims*4, embed_dims, bias=False
        )) 
                        
    def loss(self, batch_inputs, batch_data_samples, train_cfg):
        class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds = self.forward(
            batch_inputs, batch_data_samples)
        
        cls_targets_buffer, mask_targets_buffer, label_weights_buffer = self.bipartite_matching(class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, batch_data_samples)
        losses = dict()
        for i in range(self.num_decoder_layers+1):
            losses.update(self.loss_single_layer(class_preds_buffer[i], mask_preds_buffer[i], pos_mask_preds_buffer[i],
                                                cls_targets_buffer[i], mask_targets_buffer[i], label_weights_buffer[i], i))
        
        # semantic loss
        if self.use_sem_loss:
            gt_semantic_segs = [
                data_sample.gt_pts_seg.voxel_semantic_mask
                for data_sample in batch_data_samples
            ]
            seg_label = torch.cat(gt_semantic_segs)
            sem_preds = torch.cat(sem_preds, dim=0)
            
            losses['loss_ce'] = self.loss_ce(
                sem_preds, seg_label, ignore_index=self.ignore_index) 
            losses['loss_lovasz'] = self.loss_lovasz(
                sem_preds, seg_label, ignore_index=self.ignore_index) 
        
        # heatmap loss
        pred_heatmap = batch_inputs['pred_heatmap'].view(-1, self.grid_size[0], self.grid_size[1])
        gt_heatmap = self._process_gt_data(batch_inputs['voxels']['gt_heatmap_bev'], 
                                         pred_heatmap.device, len(batch_data_samples))
        losses['loss_heatmap'] = self.loss_heatmap(pred_heatmap, gt_heatmap)
            
        # fog loss
        pred_fog = batch_inputs['pred_fog'].squeeze(1)
        gt_fog = self._process_gt_data(batch_inputs['voxels']['gt_fog_bev'], 
                                     pred_fog.device, len(batch_data_samples), 
                                     unsqueeze=True, dtype=torch.float32)
        losses['loss_fog'] = self.loss_fog(pred_fog, gt_fog)
        return losses
    

    def forward(self, batch_inputs, batch_data_samples):
        
        features = batch_inputs['features']
        voxel_coors = batch_inputs['voxels']['voxel_coors']
        img_features = batch_inputs['img_features']
        batch_data_samples = batch_data_samples
        points_coords = batch_inputs['points']
        
        class_preds_buffer = []
        mask_preds_buffer = []
        pos_mask_preds_buffer = []

        batch_size = voxel_coors[:, 0].max().item() + 1        
        lidar_feature_split = []
        voxel_coor_split = [] 
        for i in range(batch_size):
            lidar_feature_split.append(features[voxel_coors[:, 0] == i])
            voxel_coor_split.append(voxel_coors[voxel_coors[:, 0] == i][:, 1:])
        
            
        # 1. apply SPE on lidar and img features
        lidar_pe_features, lidar_pe = multi_apply(self.cpe_lidar, lidar_feature_split, voxel_coor_split)
        img_pe_features, img_pe = multi_apply(self.cpe_img, img_features, lidar_pe)

        # 2. get mask_update_feature and pe
        mm_pe_features, _ = multi_apply(self.fuse_mm_features, lidar_pe_features, img_pe_features, fuse_layer=self.feat_fuse_layer)
        mm_pes, _ = multi_apply(self.fuse_mm_features, lidar_pe, img_pe, fuse_layer=self.pe_fuse_layer)
        
        # 3. get query_update_feature: semantic and instance
        semQ_feat = lidar_pe_features
        insQ_feat = mm_pe_features
        
        # 4. init queries
        queries, q_pe_embed, sem_preds = self.mm_init_query(voxel_coor_split, insQ_feat, semQ_feat, points_coords, batch_data_samples, batch_inputs)

        # 5. get mask and class predictions  
        class_preds, mask_preds, pos_mask_preds = multi_apply(self.pa_seg_single, queries, mm_pe_features, mm_pes, layer=0)
        
        class_preds_buffer.append(None)
        mask_preds_buffer.append(mask_preds)
        pos_mask_preds_buffer.append(pos_mask_preds)
        for i in range(self.num_decoder_layers):
            mm_pe_features, _ = multi_apply(self.fuse_mm_features, lidar_pe_features, img_pe_features, fuse_layer=self.feat_fuse_layer)
            mm_pes, _ = multi_apply(self.fuse_mm_features, lidar_pe, img_pe, fuse_layer=self.pe_fuse_layer)

            
            queries = self.query_update(i, queries, insQ_feat, mask_preds)
            class_preds, mask_preds, pos_mask_preds = multi_apply(self.pa_seg_single, queries, mm_pe_features, mm_pes, layer=i+1)
            class_preds_buffer.append(class_preds)
            mask_preds_buffer.append(mask_preds)
            pos_mask_preds_buffer.append(pos_mask_preds)
        return class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds
    
    def fuse_mm_features(self, lidar_pe_features, img_pe_features, fuse_layer=None):
        mm_features = fuse_layer(torch.cat([lidar_pe_features, img_pe_features], dim=1))
        return mm_features, None
       
    
    def mm_init_query(self, voxel_coor_split, insQ_feat, semQ_feat, points_coords, batch_data_samples, batch_inputs):
        """
        Initiate lidar_ and img_ feature and pe, initiate thing and stuff query.
        
        """
        
        heatmaps = [batch_inputs['pred_heatmap'][i] for i in range(len(batch_data_samples))]
        fogmaps = [batch_inputs['pred_fog'][i] for i in range(len(batch_data_samples))]
        queries_coor, queries_ind, N_valid = multi_apply(self.gen_mm_query, batch_data_samples, voxel_coor_split, heatmaps, fogmaps)
        queries_content = init_query_content(len(insQ_feat), N_valid, self.queries, insQ_feat, queries_ind)
        queries, q_pe_embed = multi_apply(self.cpe_query, queries_content, queries_coor, batch_data_samples, points_coords, voxel_coor_split)
        
        sem_preds = []
        batch_size = len(voxel_coor_split)
        if self.use_sem_loss:
            sem_queries = self.sem_queries.weight.clone().squeeze(-1).squeeze(-1).repeat(1,1,batch_size).permute(2,0,1) # [B, num_classes, embed_dims]
            for b in range(len(semQ_feat)):
                sem_pred = torch.einsum("nc,vc->vn", sem_queries[b], semQ_feat[b]) 
                sem_preds.append(sem_pred)
                
                stuff_queries = sem_queries[b][self.stuff_class] # split stuff query from semantic query
                queries[b] = torch.cat([queries[b], stuff_queries], dim=0)

        return queries, q_pe_embed, sem_preds

    def cpe_query(self, query_content, queries_coor, data_sample=None, point_coor=None, voxel_coor=None):
        layers = {'polar_proj': self.polar_proj_q, 'polar_norm': self.polar_norm_q, 
                    'cart_proj': self.cart_proj_q, 'cart_norm': self.cart_norm_q,
                    'pe_conv': self.pe_conv_q, 
                    'position_proj': self.position_proj_q, 'position_norm': self.position_norm_q,
                    'feat_conv': self.feat_conv_q}
        if len(queries_coor.shape) < 2:
            queries_coor = torch.zeros(0, 3).to(queries_coor.device)
        
        assert queries_coor.shape[1] == 3, 'input shape should be [N, 3]'
        
        center_polar, center_cart, scale = get_center_scale_voxel(queries_coor, self.grid_size, 
                                                                    self.point_cloud_range_polar, self.point_cloud_range_cart,
                                                                    ret_scale=True)
        _, cent_pe = self.mpe_layer_single(center_polar, center_cart, layers, None, aggregate_mode=None)
        scale_pe = self.fc_coor_scale_q(scale.reshape(1, -1, 1)).reshape(8, -1, 256) # [8, N, 256]
        scale_pe = self.pe_aggregate(scale_pe, aggregate_mode='avg') # [N, 256]
        final_pe = cent_pe + scale_pe
        final_pe = torch.cat([final_pe, torch.zeros(self.num_queries-final_pe.shape[0], final_pe.shape[1]).to(final_pe.device)], dim=0)
        query_final = query_content + final_pe   
        return query_final, final_pe

    
    def gen_mm_query(self, data_sample, voxel_coor, pred_heatmap, pred_fog):
        """
        Get the valid queries' coor and their corresponding indices.
        """        
        # $$$$$$$$$$$$$$$$$$$$$$$$ USE 2D INST QUERY $$$$$$$$$$$$$$$$$$$$$$$$
        cand_vcoor_2d, cand_vscore_2d, ind_2d = select_2DQ(voxel_coor, 
                                                           data_sample.cand_2d['sam_vmask_2d'], 
                                                           data_sample.cand_2d['sam_vscore_2d'], 
                                                           pred_fog, self.grid_size)
        
        cand_vcoor_3d, cand_vscore_3d, ind_3d = select_3DQ(pred_heatmap, voxel_coor, nms_para=self.inst_3d_params['nms_para_3d'])
        cand_vscore_2d = (cand_vscore_2d - cand_vscore_2d.min())/(cand_vscore_2d.max()-cand_vscore_2d.min())
        cand_vscore_3d  = (cand_vscore_3d - cand_vscore_3d.min())/(cand_vscore_3d.max()-cand_vscore_3d.min())
        
        fused_vcoor = torch.cat([cand_vcoor_3d, cand_vcoor_2d], dim=0)
        fused_score = torch.cat([cand_vscore_3d, cand_vscore_2d], dim=0)
        ind_3d = ind_3d.to(fused_vcoor.device)
        ind_fuse = torch.cat([ind_3d, ind_2d], dim=0)
        assert fused_vcoor.shape[0]==fused_score.shape[0], 'fused_vcoor and fused_score should have the same length'
        assert fused_vcoor.shape[0]==ind_fuse.shape[0], 'fused_vcoor and ind_fuse should have the same length'
        
        select_cand, select_cand_ind = fps_sample(fused_vcoor, self.inst_3d_params['nms_para_fuse']['K'])
        ind = ind_fuse[select_cand_ind]
        
        N_valid = select_cand.shape[0]
        return select_cand, ind, N_valid
    
    def with_pos_embed(self, tensor, pos, split_query=False):
        if pos is None:
            return tensor, None
        if split_query:
            assert tensor.shape[0]>=pos.shape[0], 'tensor should have more elements than pos'
            pos = torch.cat([pos, torch.zeros(tensor.shape[0]-pos.shape[0], pos.shape[1]).to(pos.device)], dim=0)
            
        return tensor + pos, pos
    
    
    def predict(self, batch_inputs, batch_data_samples):
        class_preds_buffer, mask_preds_buffer, _, sem_preds = self.forward(batch_inputs, batch_data_samples)
        semantic_preds, instance_ids = self.generate_panoptic_results(class_preds_buffer[-1], mask_preds_buffer[-1])
                    
        pts_semantic_preds = []
        pts_instance_preds = []

        for batch_idx in range(len(batch_data_samples)):
            semantic_sample = semantic_preds[batch_idx]
            instance_sample = instance_ids[batch_idx]
            point2voxel_map = batch_data_samples[
                batch_idx].gt_pts_seg.point2voxel_map.long()
            point_semantic_sample = semantic_sample[point2voxel_map]
            point_instance_sample = instance_sample[point2voxel_map]
            pts_semantic_preds.append(point_semantic_sample.cpu().numpy())
            pts_instance_preds.append(point_instance_sample.cpu().numpy())

        return pts_semantic_preds, pts_instance_preds
   
    def cpe_lidar(self, lidar_feature, voxel_coor):
        """Encode 3D features with sparse indices: centeriod and extreme points.
        Params:
            features: [[N, C], ...]
            voxel_coor: [[N, 4], ...], 4: batch_idx, x, y, z
        Return:
            features: [[N, C], ...]
            normed_polar_coors: [[N, 3], ...]
            normed_cart_coors: [[N, 3], ...]
        """
        layers = {'polar_proj': self.polar_proj, 'polar_norm': self.polar_norm, 
                  'cart_proj': self.cart_proj, 'cart_norm': self.cart_norm,
                  'pe_conv': self.pe_conv, 
                  'position_proj': self.position_proj, 'position_norm': self.position_norm, 
                  'feat_conv': self.feat_conv}  
        center_polar, center_cart, scale = get_center_scale_voxel(voxel_coor, self.grid_size, 
                                                                    self.point_cloud_range_polar, self.point_cloud_range_cart,
                                                                    ret_scale=True)
        _, cent_pe = self.mpe_layer_single(center_polar, center_cart, layers, None, aggregate_mode=None)
        scale_pe = self.fc_coor_scale_lidar(scale.view(1, -1, 1)).view(8, -1, 256) # [8, N, 256]
        scale_pe = self.pe_aggregate(scale_pe, aggregate_mode='avg') # [N, 256]
        final_pe = cent_pe + scale_pe
        
        lidar_pe_feature = lidar_feature + final_pe   
        return lidar_pe_feature, final_pe
     
    def cpe_img(self, img_feature, lidar_pe=None):
        img_pe = lidar_pe
        img_pe_feature = img_feature + img_pe
        return img_pe_feature, img_pe
        
        
    def query_update(self, layer, queries, input_features, mask_preds):
        """
        Pack query_update procedures together
        """
        queries = self.transformer_decoder[layer](queries, input_features, mask_preds)
        return queries
        
    def mask_update(self, queries, lidar_features, lidar_pe, img_features, img_pe, layer):
        if lidar_pe is None:
            lidar_pe = [None] * len(lidar_features)
        class_preds, mask_preds, pos_mask_preds = multi_apply(
            self.mask_update_single, queries, lidar_features, lidar_pe, img_features, img_pe, [layer] * len(lidar_features))
        return class_preds, mask_preds, pos_mask_preds


    def mask_update_single(self, queries, features, mpe, img_pe_features, img_pe, layer):
        """Get Predictions of a single sample level."""
        mm_features = self.mm_conv(torch.cat([features, img_pe_features], dim=1)) #~100MB
        mm_pes = self.mm_conv(torch.cat([mpe, img_pe], dim=1))
        
        mask_queries = queries
        mask_queries = self.fc_mask[layer](mask_queries)
        mask_pred = torch.einsum('nc,vc->nv', mask_queries, mm_features)

        if self.use_pa_seg:
            pos_mask_queries = queries
            pos_mask_queries = self.fc_coor_mask[layer](pos_mask_queries)
            pos_mask_pred = torch.einsum('nc,vc->nv', pos_mask_queries, mm_pes)
            mask_pred = mask_pred + pos_mask_pred
        else:
            pos_mask_pred = None

        if layer != 0:
            cls_queries = queries
            cls_pred = self.fc_cls[layer](cls_queries)
        else:
            cls_pred = None

        return cls_pred, mask_pred, pos_mask_pred
    
       
    def mpe_layer_single(self, normed_polar_coor, normed_cart_coor, layers, feature=None, aggregate_mode=None):
        """ 1. Project normed polar and cartisian coordinates to embedding space for every frame. 
            2. Employ same projection on features if feature is not None.
            3. Aggregate along the first channel if required.
            
        Params:
            normed_polar_coors: [N, 3] or [B,N,3]
            normed_cart_coors: [[N, 3] or [B,N,3]
            feature: [N, C] or [B,N,3]
        Return:
            polar_pe: [N, C] or [B,N,3]
            cart_pe: [N, C] or [B,N,3]
        """
        assert normed_polar_coor.shape[-1] == 3, "Coordinate should be 3D!"
        assert normed_cart_coor.shape[-1] == 3, "Coordinate should be 3D!"
        
        if self.pe_type == 'polar' or self.pe_type == 'cart':
            pe = layers['position_norm'](
                    layers['position_proj'](normed_polar_coor.float()))
            if aggregate_mode is not None:
                pe = self.pe_aggregate(pe, aggregate_mode) # [N, 256]
            feature = feature + pe
            feature = layers['feat_conv'](feature)
                    
        elif self.pe_type == 'mpe':
            cart_pe = layers['cart_norm'](
                layers['cart_proj'](normed_cart_coor.float()))
            polar_pe = layers['polar_norm'](
                layers['polar_proj'](normed_polar_coor.float()))
            for pc in layers['pe_conv']:
                polar_pe = pc(polar_pe)
                cart_pe = pc(cart_pe)
                if feature is not None:
                    feature = pc(feature)
            pe = cart_pe + polar_pe
            if aggregate_mode is not None:
               pe = self.pe_aggregate(pe, aggregate_mode) # [N, 256]
            feature = feature + pe if feature is not None else None
        return feature, pe
         
    def pe_aggregate(self, pe, aggregate_mode):
        if aggregate_mode == 'max':
            pe_ = pe.max(dim=0).values
        elif aggregate_mode == 'avg':
            pe_ = pe.mean(dim=0)
        elif aggregate_mode == 'set':
            pe_max = pe.max(dim=0).values
            pe_avg = pe.mean(dim=0)
            pe_ = 0.5*pe_max + 0.5*pe_avg
        return pe_.squeeze(0)
    
    def _process_gt_data(self, gt_data_list, target_device, batch_size, 
                        unsqueeze=False, dtype=None):
        processed_data = []
        for i in range(batch_size):
            data = torch.from_numpy(gt_data_list[i]).to(target_device)
            if unsqueeze:
                data = data.unsqueeze(0)
            if dtype is not None:
                data = data.to(dtype)
            processed_data.append(data)
        return torch.cat(processed_data, dim=0)
          
def pack_3d_query(hm_candidates):
    hm_candidates_final = hm_candidates.flatten()
    return hm_candidates_final

def init_query_content(batch_size, num_valid, query_layer, insQ_features, queries_ind):
    query_content = []
    pad_pc = query_layer.weight.clone().squeeze(0).squeeze(0).repeat(batch_size,1,1).permute(0,2,1) # [B, num_queries, embed_dims]
    pad_pc = [pad_pc[i] for i in range(batch_size)]
    for i in range(batch_size):
        N_valid = num_valid[i]
        qc = extract_selected_features(queries_ind[i][:N_valid].long(), insQ_features[i])
        qc = torch.cat([qc, pad_pc[i][N_valid:]], dim=0)
        query_content.append(qc)
    return query_content