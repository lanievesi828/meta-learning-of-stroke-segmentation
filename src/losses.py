# 文件: src/losses_v2.py (一个新文件，或者直接修改旧的)

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# get_laplacian_edge_mask 函数保持不变
def get_laplacian_edge_mask(target_gt):
    kernel = torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], 
                        dtype=torch.float32, device=target_gt.device).view(1,1,3,3)
    edges = F.conv2d(target_gt.float(), kernel, padding=1).abs()
    return (edges > 0).float()

# --- 这是新的、更忠于原始逻辑的MAML兼容损失 ---

class MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss(nn.Module):
    """
    【MAML兼容最终版】
    此损失函数在标准训练模式下（reduction='mean'）
    其Dice部分的计算逻辑与原始的批次聚合损失函数在数学上完全等价。
    同时，它支持 'none' 模式，为MAML的内循环提供逐样本损失。
    """
    def __init__(self, bce_weight=0.1, edge_weight=3.0, smooth=1e-8, squared_denom=False):
        super().__init__()
        self.bce_weight = bce_weight
        self.edge_weight = edge_weight
        self.smooth = smooth
        self.squared_denom = squared_denom
        # BCE损失仍然可以逐样本计算，因为它是线性可加的
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs_logits, targets_gt, reduction='mean'):
        B = inputs_logits.size(0)
        pred_probs = torch.sigmoid(inputs_logits)

        # 1. 计算BCE损失 (这部分和之前一样，是正确的)
        bce_loss_per_sample = self.bce_loss_fn(inputs_logits, targets_gt.float()).view(B, -1).mean(1)

        # 2. 计算Dice损失 (核心修改部分)
        
        # 展平输入，但保留批次维度
        pred_flat = pred_probs.view(B, -1)
        target_flat = targets_gt.view(B, -1)
        factor = 2 if self.squared_denom else 1

        # --- Base Dice 部分 ---
        # 计算每个样本的交集和分母项
        base_intersection_per_sample = (pred_flat * target_flat).sum(1)
        base_denom_per_sample = pred_flat.pow(factor).sum(1) + target_flat.pow(factor).sum(1)

        # --- Edge Dice 部分 ---
        edge_mask = get_laplacian_edge_mask(targets_gt)
        pred_edges_flat = (pred_probs * edge_mask).view(B, -1)
        target_edges_flat = (targets_gt * edge_mask).view(B, -1)
        
        edge_intersection_per_sample = (pred_edges_flat * target_edges_flat).sum(1)
        edge_denom_per_sample = pred_edges_flat.pow(factor).sum(1) + target_edges_flat.pow(factor).sum(1)

        # 3. 根据 reduction 参数进行聚合
        if reduction == 'mean':
            # --- 模拟原始的批次聚合行为 ---
            # a. 聚合所有样本的交集和分母项
            total_base_intersection = base_intersection_per_sample.sum()
            total_base_denom = base_denom_per_sample.sum()
            
            total_edge_intersection = edge_intersection_per_sample.sum()
            total_edge_denom = edge_denom_per_sample.sum()

            # b. 用聚合后的值计算整个批次的Dice损失
            base_dice_loss = 1 - (2. * total_base_intersection + self.smooth) / (total_base_denom + self.smooth)
            edge_dice_loss = 1 - (2. * total_edge_intersection + self.smooth) / (total_edge_denom + self.smooth)
            
            dice_loss = base_dice_loss + self.edge_weight * edge_dice_loss
            
            # c. BCE损失是可加的，所以可以直接取均值
            bce_loss = bce_loss_per_sample.mean()
            
            # d. 最终组合损失
            combined_loss = self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
            return combined_loss

        elif reduction == 'none':
            # --- 为MAML内循环提供逐样本损失 ---
            # 这种模式下，我们只能使用逐样本计算的方式，这与原始逻辑不等价，但对MAML是必要的
            base_dice_loss_per_sample = 1 - (2. * base_intersection_per_sample + self.smooth) / (base_denom_per_sample + self.smooth)
            edge_dice_loss_per_sample = 1 - (2. * edge_intersection_per_sample + self.smooth) / (edge_denom_per_sample + self.smooth)
            
            dice_loss_per_sample = base_dice_loss_per_sample + self.edge_weight * edge_dice_loss_per_sample
            
            combined_loss_per_sample = self.bce_weight * bce_loss_per_sample + (1.0 - self.bce_weight) * dice_loss_per_sample
            return combined_loss_per_sample
            
        else:
            raise ValueError(f"不支持的 reduction 模式: {reduction}")