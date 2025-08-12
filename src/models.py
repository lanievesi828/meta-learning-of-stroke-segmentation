# 文件: src/models.py
# 版本：最终移植版，旨在精确复刻旧的高性能模型，并为其添加MAML兼容性。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
import sys

# ==============================================================================
# 基础模块 (移植自旧模型，并添加MAML兼容性)
# ==============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # 完全复刻旧模型的Sequential结构和默认的bias=True
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, params=None, prefix=''):
        if params is None:
            return self.double_conv(x)
        else:
            # 手动、精确地展开Sequential的函数式调用
            # Conv 1
            x = F.conv2d(x, params[f'{prefix}double_conv.0.weight'], params[f'{prefix}double_conv.0.bias'], padding=1)
            # BN 1
            x = F.batch_norm(x, self.double_conv[1].running_mean, self.double_conv[1].running_var, 
                             params[f'{prefix}double_conv.1.weight'], params[f'{prefix}double_conv.1.bias'], 
                             training=True, eps=self.double_conv[1].eps, momentum=self.double_conv[1].momentum)
            # ReLU 1
            x = F.relu(x, inplace=True)
            
            # Conv 2
            x = F.conv2d(x, params[f'{prefix}double_conv.3.weight'], params[f'{prefix}double_conv.3.bias'], padding=1)
            # BN 2
            x = F.batch_norm(x, self.double_conv[4].running_mean, self.double_conv[4].running_var, 
                             params[f'{prefix}double_conv.4.weight'], params[f'{prefix}double_conv.4.bias'], 
                             training=True, eps=self.double_conv[4].eps, momentum=self.double_conv[4].momentum)
            # ReLU 2
            x = F.relu(x, inplace=True)
            return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x, params=None, prefix=''):
        if params is None:
            return self.maxpool_conv(x)
        else:
            x = self.maxpool_conv[0](x) # MaxPool无参数
            # 为内部的DoubleConv传递正确的prefix。因为DoubleConv在Sequential中是第2个元素（索引为1）
            # 所以它的参数名前缀会是 'maxpool_conv.1.'
            double_conv_prefix = f'{prefix}maxpool_conv.1.'
            x = self.maxpool_conv[1](x, params, prefix=double_conv_prefix)
            return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, params=None, prefix=''):
        if params is None:
            return self.conv(x)
        else:
            return F.conv2d(x, params[f'{prefix}conv.weight'], params[f'{prefix}conv.bias'], padding=0)

# ==============================================================================
# Transformer 模块 (完全移植自旧模型，并添加MAML兼容性)
# ==============================================================================

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError("Bias not implemented in the original model.")
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, params=None, prefix=''):
        b, wh, d = x.size()
        weight = params[f'{prefix}weight'] if params is not None else self.weight
        # torch.bmm 要求批次维度匹配，所以需要repeat
        x = torch.bmm(x, weight.repeat(b, 1, 1))
        return x
        
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000**(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, tensor): # 输入 (B, H, W, C)
        if len(tensor.shape) != 4: raise RuntimeError("The input tensor has to be 4d!")
        b, x_dim, y_dim, orig_ch = tensor.shape
        pos_x = torch.arange(x_dim, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y_dim, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x_dim, y_dim, self.channels * 2), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y
        return emb[None, :, :, :orig_ch].repeat(b, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)
    
    def forward(self, tensor): # 输入 (B, C, H, W)
        tensor_permuted = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor_permuted)
        return enc.permute(0, 3, 1, 2)
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1) # 旧模型是在dim=1上
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x, params=None, prefix=''):
        b, c, h, w = x.size()
        pe = self.pe(x) # PE无参数，直接调用
        x_pe = x + pe
        
        x_reshaped = x_pe.view(b, c, h * w).permute(0, 2, 1)
        
        if params is None:
            Q = self.query(x_reshaped)
            K = self.key(x_reshaped)
            V = self.value(x_reshaped)
        else:
            Q = self.query(x_reshaped, params, prefix=f'{prefix}query.')
            K = self.key(x_reshaped, params, prefix=f'{prefix}key.')
            V = self.value(x_reshaped, params, prefix=f'{prefix}value.')
            
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))
        x_out = torch.bmm(A, V).permute(0, 2, 1).view(b, c, h, w)
        return x_out

# --- BiFPN 和 ChannelAdjustment (移植自旧模型，并添加MAML兼容性) ---
# ... 此处省略，请确保你旧脚本中对应的类定义被完整复制到这里 ...
# 为了保证完整性，我将提供一个改造后的版本
class WeightedFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_inputs=2, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        
    def forward(self, *xs, params=None, prefix=''):
        weights = params[f'{prefix}weights'] if params is not None else self.weights
        norm_weights = F.relu(weights)
        norm_weights = norm_weights / (torch.sum(norm_weights, dim=0) + self.epsilon)
        
        fused_feature = sum(norm_weights[i] * xs[i] for i in range(len(xs)))
        
        if params is None:
            return self.conv(fused_feature)
        else:
            x = F.conv2d(fused_feature, params[f'{prefix}conv.0.weight'], params[f'{prefix}conv.0.bias'], padding=1)
            x = F.batch_norm(x, self.conv[1].running_mean, self.conv[1].running_var, 
                             params[f'{prefix}conv.1.weight'], params[f'{prefix}conv.1.bias'], training=True)
            return F.relu(x, inplace=True)

class BiFPNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.td_fusion_1 = WeightedFeatureFusion(channels, channels)
        self.td_fusion_2 = WeightedFeatureFusion(channels, channels)
        self.td_fusion_3 = WeightedFeatureFusion(channels, channels)
        self.bu_fusion_1 = WeightedFeatureFusion(channels, channels)
        self.bu_fusion_2 = WeightedFeatureFusion(channels, channels)
        self.bu_fusion_3 = WeightedFeatureFusion(channels, channels)
        self.upsample_4_to_3 = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.upsample_3_to_2 = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.upsample_2_to_1 = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.downsample_1_to_2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.downsample_2_to_3 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.downsample_3_to_4 = nn.Conv2d(channels, channels, 3, 2, 1)

    def _apply_transpose_conv(self, x, params, prefix):
        return F.conv_transpose2d(x, params[f'{prefix}.weight'], params.get(f'{prefix}.bias'), stride=2)
    def _apply_conv(self, x, params, prefix):
        return F.conv2d(x, params[f'{prefix}.weight'], params.get(f'{prefix}.bias'), stride=2, padding=1)
    def _interpolate(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False) if x.shape[2:] != target.shape[2:] else x

    def forward(self, p1, p2, p3, p4, params=None, prefix=''):
        # Top-down
        up4 = self.upsample_4_to_3(p4) if params is None else self._apply_transpose_conv(p4, params, f'{prefix}upsample_4_to_3')
        p3_td = self.td_fusion_1(p3, self._interpolate(up4, p3), params=params, prefix=f'{prefix}td_fusion_1.')
        up3 = self.upsample_3_to_2(p3_td) if params is None else self._apply_transpose_conv(p3_td, params, f'{prefix}upsample_3_to_2')
        p2_td = self.td_fusion_2(p2, self._interpolate(up3, p2), params=params, prefix=f'{prefix}td_fusion_2.')
        up2 = self.upsample_2_to_1(p2_td) if params is None else self._apply_transpose_conv(p2_td, params, f'{prefix}upsample_2_to_1')
        p1_out = self.td_fusion_3(p1, self._interpolate(up2, p1), params=params, prefix=f'{prefix}td_fusion_3.')
        # Bottom-up
        down1 = self.downsample_1_to_2(p1_out) if params is None else self._apply_conv(p1_out, params, f'{prefix}downsample_1_to_2')
        p2_out = self.bu_fusion_1(p2_td, self._interpolate(down1, p2_td), params=params, prefix=f'{prefix}bu_fusion_1.')
        down2 = self.downsample_2_to_3(p2_out) if params is None else self._apply_conv(p2_out, params, f'{prefix}downsample_2_to_3')
        p3_out = self.bu_fusion_2(p3_td, self._interpolate(down2, p3_td), params=params, prefix=f'{prefix}bu_fusion_2.')
        down3 = self.downsample_3_to_4(p3_out) if params is None else self._apply_conv(p3_out, params, f'{prefix}downsample_3_to_4')
        p4_out = self.bu_fusion_3(p4, self._interpolate(down3, p4), params=params, prefix=f'{prefix}bu_fusion_3.')
        return p1_out, p2_out, p3_out, p4_out

class ChannelAdjustment(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, params=None, prefix=''):
        if params is None: return self.relu(self.bn(self.conv(x)))
        x = F.conv2d(x, params[f'{prefix}conv.weight'], params.get(f'{prefix}conv.bias'))
        x = F.batch_norm(x, self.bn.running_mean, self.bn.running_var, params[f'{prefix}bn.weight'], params[f'{prefix}bn.bias'], training=True)
        return F.relu(x, inplace=True)


# ==============================================================================
# 主模型 (最终统一版本，命名为 MAML_BiFPN_Transformer_UNet 以便你的脚本直接使用)
# ==============================================================================

class MAML_BiFPN_Transformer_UNet(nn.Module):
    def __init__(self, in_channels=1, classes=1, bifpn_channels=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.MHSA = MultiHeadSelfAttention(512)
        self.adjust_p1 = ChannelAdjustment(64, bifpn_channels)
        self.adjust_p2 = ChannelAdjustment(128, bifpn_channels)
        self.adjust_p3 = ChannelAdjustment(256, bifpn_channels)
        self.adjust_p4 = ChannelAdjustment(512, bifpn_channels)
        self.bifpn = BiFPNBlock(bifpn_channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(bifpn_channels, bifpn_channels, 3, 1, 1),
            nn.BatchNorm2d(bifpn_channels),
            nn.ReLU(inplace=True)
        )
        self.outc = OutConv(bifpn_channels, classes)
        
    def forward(self, x, params=None):
        if params is None:
            # 标准监督学习模式 (与旧脚本完全一致)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x4_att = self.MHSA(x4)
            p1 = self.adjust_p1(x1)
            p2 = self.adjust_p2(x2)
            p3 = self.adjust_p3(x3)
            p4 = self.adjust_p4(x4_att)
            p1_out, _, _, _ = self.bifpn(p1, p2, p3, p4)
            final_features = self.final_conv(p1_out)
            logits = self.outc(final_features)
            return logits
        else:
            # MAML 函数式调用模式
            prefix = '' # 假设顶层没有prefix
            x1 = self.inc(x, params, prefix=f'{prefix}inc.')
            x2 = self.down1(x1, params, prefix=f'{prefix}down1.')
            x3 = self.down2(x2, params, prefix=f'{prefix}down2.')
            x4 = self.down3(x3, params, prefix=f'{prefix}down3.')
            
            x4_att = self.MHSA(x4, params, prefix=f'{prefix}MHSA.')
            
            p1 = self.adjust_p1(x1, params, prefix=f'{prefix}adjust_p1.')
            p2 = self.adjust_p2(x2, params, prefix=f'{prefix}adjust_p2.')
            p3 = self.adjust_p3(x3, params, prefix=f'{prefix}adjust_p3.')
            p4 = self.adjust_p4(x4_att, params, prefix=f'{prefix}adjust_p4.')
            
            p1_out, _, _, _ = self.bifpn(p1, p2, p3, p4, params=params, prefix=f'{prefix}bifpn.')
            
            # 函数式 final_conv
            x = F.conv2d(p1_out, params[f'{prefix}final_conv.0.weight'], params[f'{prefix}final_conv.0.bias'], padding=1)
            x = F.batch_norm(x, self.final_conv[1].running_mean, self.final_conv[1].running_var,
                             params[f'{prefix}final_conv.1.weight'], params[f'{prefix}final_conv.1.bias'], training=True)
            final_features = F.relu(x, inplace=True)
            
            logits = self.outc(final_features, params, prefix=f'{prefix}outc.')
            return logits