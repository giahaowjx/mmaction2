# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.utils import _BatchNorm
from .misc_head import ACRNHead

try:
    from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False
  
# Note: All these heads take 5D Tensors as input (N, C, T, H, W)


class HR2O_NL(nn.Module):
    def __init__(out_channels=512, kernel_size=3, mlp_1x1=False, conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='GN3d', requires_grad=True, num_features=out_channels, num_groups=1,
                               affine=True)):
        super(HR2O_NL, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        padding = kernel_size // 2
        self.conv_q = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False,
            conv_cfg=conv_cfg)
        
        self.conv_k = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False,
            conv_cfg=conv_cfg)
        
        self.conv_v = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False,
            conv_cfg=conv_cfg)
        
        self.kernel_size = 1 if mlp_1x1 else kernel_size
        
        self.conv = ConvModule(
            out_channel,
            out_channels,
            kernel_size=(1, self.kernel_size, self.kernel_size),
            padding=(0, padding, padding),
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        
        
    def forward(x, rois):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        value = self.conv_v(x)
        virt_feats = torch.zeros_like(value)
        roi_inds = rois[:, 0].type(torch.long)
        
        for roi_ind in set(roi_inds):
            inds = (roi_inds == roi_ind)
            att = (query[inds] * key[inds]).sum(2) / (self.out_channels ** 0.5)
            att = nn.Softmax(dim=1)(att)
            virt_feats[inds] = (att.unsqueeze(2) * value[inds]).sum(1)
        
        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        
        x = x + virt_feats
        return x
        


class ACARHead(nn.Module):
    """ACAR Head: Tile + 1x1 convolution + 3x3 convolution + HR2O Module

    This module is proposed in
    `Actor-Context-Actor Relation Network for Spatio-Temporal Action Localization Junting
    <https://arxiv.org/pdf/1807.10982v1.pdf>`_

    Args:
        in_channels (int): The input channel.
        out_channels (int): The output channel.
        stride (int): The spatial stride.
        num_convs (int): The number of 3x3 convolutions in ACRNHead.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        kwargs (dict): Other new arguments, to be compatible with MMDet update.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 num_convs=1,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 **kwargs):
        
        self.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        self.acrn_head = ACRNHead(in_channels, out_channels, stride, num_convs, conv_cfg, norm_cfg, act_cfg, **kwargs)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.hr2o = HR2O_NL()

        
    def init_weights(self, **kwargs):
        """Weight Initialization for ACRNHead."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x, feat, rois, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The extracted RoI feature.
            feat (torch.Tensor): The context feature.
            rois (torch.Tensor): The regions of interest.

        Returns:
            torch.Tensor: The RoI features that have interacted with context
                feature.
        """
        # We use max pooling by default
        x = self.acrn_head(x, feat, rois, **kwargs)
        x = self.max_pool(x)
        x = self.hr2o(x, rois)

        return new_feat
        
        
if mmdet_imported:
    MMDET_SHARED_HEADS.register_module()(ACARHead)