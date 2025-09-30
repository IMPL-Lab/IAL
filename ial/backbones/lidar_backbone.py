from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from mmcv.ops import SparseConvTensor, SparseMaxPool3d

from .cylinder3d import _Asymm3DSpconv
from tools.general_tool import clone_detach_SparseConvTensor

@MODELS.register_module(force=True)
class Cylinder3D_HM(_Asymm3DSpconv):
    """
    Cylinder3D backbone with heatmap and fog branches.
    """

    def __init__(self,
                 grid_size: int,
                 input_channels: int,
                 base_channels: int = 16,
                 backbone_depth: int = 4,
                 height_pooing: List[bool] = [True, True, False, False],
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg=None,
                 more_conv=False,
                 out_channels=128):
        super(Cylinder3D_HM, self).__init__(grid_size=grid_size,
                        input_channels=input_channels,
                        base_channels=base_channels,
                        backbone_depth=backbone_depth,
                        height_pooing=height_pooing,
                        norm_cfg=norm_cfg,
                        init_cfg=init_cfg,
                        more_conv=more_conv,
                        out_channels=out_channels)
        
        in_channels= 256
        num_classes = 1 
        num_heatmap_convs = 2
        share_conv_channel = 2 * base_channels 
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3)
        self.shared_conv = shared_conv_circular(in_channels, share_conv_channel, dilation=1)
        self.pool3d_heatmap = SparseMaxPool3d((1, 1, 32))
    
        heads = dict(heatmap=(num_classes, num_heatmap_convs))
        separate_head.update(
            in_channels=share_conv_channel, heads=heads, num_cls=num_classes)
        self.heatmap_heads = MODELS.build(separate_head)
            
        
        self.pool3d_fog = SparseMaxPool3d((1, 1, 32)) # follows TransFusion
        self.shared_conv_fog = shared_conv_circular(in_channels, share_conv_channel, dilation=1)
        heads_fog = dict(fog=(1, num_heatmap_convs)) 
        separate_head_fog=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3)
        separate_head_fog.update(
            in_channels=share_conv_channel, heads=heads_fog, num_cls=num_classes)
        self.fog_heads = MODELS.build(separate_head_fog)
                
    def forward(self, voxel_features: torch.Tensor, coors: torch.Tensor,
                batch_size: int, batch_data_samples) -> SparseConvTensor:
        ret = SparseConvTensor(voxel_features, coors, np.array(self.grid_size),
                               batch_size)
        ret = self.down_context(ret)

        down_skip_list = []
        down_pool = ret
        for i in range(self.backbone_depth):
            down_pool, down_skip = self.down_block_list[i](down_pool)
            down_skip_list.append(down_skip)

        up = down_pool
        for i in range(self.backbone_depth - 1, -1, -1):
            up = self.up_block_list[i](up, down_skip_list[i])

        ddcm = self.ddcm(up)
        ddcm.features = torch.cat((ddcm.features, up.features), 1)

        if self.more_conv:
            ddcm = self.addConv(ddcm)
            ddcm.features = self.addBn(ddcm.features)
            ddcm.features = self.addAct(ddcm.features)
            
        ddcm_hm = clone_detach_SparseConvTensor(ddcm)
        infeat = self.pool3d_heatmap(ddcm_hm).dense().squeeze(-1)
        heatmap = self.shared_conv(infeat)
        heatmap = self.heatmap_heads(heatmap)['heatmap']
        
        ddcm_fog = clone_detach_SparseConvTensor(ddcm)
        infeat = self.pool3d_fog(ddcm_fog).dense().squeeze(-1)
        fog = self.shared_conv_fog(infeat)
        fog = self.fog_heads(fog)['fog']
        
        return ddcm, heatmap, fog


class shared_conv_circular(nn.Module):
    """
    Shared convolutional layers for heatmap and fog branches.
    3 layers of Conv Blocks with circular padding.
    """
    def __init__(self, in_ch, out_ch, dilation):
        super(shared_conv_circular, self).__init__()
        self.conv = nn.Sequential(
            tri_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class tri_conv_circular(nn.Module):
    def __init__(self, in_ch, out_ch,group_conv, dilation=1):
        super(tri_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv3(x)
        return x
