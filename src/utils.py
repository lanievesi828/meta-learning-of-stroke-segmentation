import torch
import random
import numpy as np
import os
import sys
import torch.nn.functional as F # 需要 F 来实现 get_edge_mask_tensor

def set_seed(seed=42):
    """设置随机种子以保证实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    # 以下两行确保cuDNN的确定性行为，可能会稍微降低速度
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已设置为: {seed}")

def dice_coefficient(pred_probs, targets, smooth=1e-6):
    """
    计算 Dice 系数。
    Args:
        pred_probs (torch.Tensor): 模型输出的概率图 (sigmoid激活后)，形状如 (B, C, H, W) 或 (H, W)。
        targets (torch.Tensor): 真实标签 (0或1)，形状与 pred_probs 相同。
        smooth (float): 防止分母为0的小常数。
    Returns:
        torch.Tensor: Dice 系数值。
    """
    pred_flat = pred_probs.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1).float() # 确保是浮点型

    intersection = (pred_flat * targets_flat).sum()
    union = pred_flat.sum() + targets_flat.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_score(pred_binary, targets_binary, smooth=1e-6):
    """
    计算 IoU (Intersection over Union) / Jaccard 系数。
    这个版本使用数值计算，更稳健。
    Args:
        pred_binary (torch.Tensor): 二值化的预测掩码 (0或1)，形状如 (B, C, H, W) 或 (H, W)。
        targets_binary (torch.Tensor): 二值化的真实标签 (0或1)，形状与 pred_binary 相同。
    Returns:
        torch.Tensor: IoU 分数值。
    """
    # 确保输入是浮点数以便进行乘法和加法
    # 并将 pred_binary 转换为 0/1 的浮点数
    pred_flat = pred_binary.contiguous().view(-1).float()
    targets_flat = targets_binary.contiguous().view(-1).float()

    # --- 核心修改 ---
    # Intersection: A * B
    # Union: A + B - (A * B)
    intersection = (pred_flat * targets_flat).sum()
    total_sum = pred_flat.sum() + targets_flat.sum()
    union = total_sum - intersection 
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def get_edge_mask_tensor(target_tensor, threshold=0.1):
    """
    为一批目标掩码生成边缘掩码。
    Args:
        target_tensor (torch.Tensor): 一批目标掩码 (B, 1, H, W)，值在0到1之间。
        threshold (float): 用于二值化边缘强度的阈值。
    Returns:
        torch.Tensor: 边缘掩码 (B, 1, H, W)，值为0或1。
    """
    # 定义 Sobel 算子用于边缘检测
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=target_tensor.device).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=target_tensor.device).unsqueeze(0).unsqueeze(0)

    # 确保 target_tensor 是浮点型并且有梯度（如果需要的话，但这里通常不需要）
    target_float = target_tensor.float()

    # 计算x和y方向的梯度
    edges_x = F.conv2d(target_float, kernel_x, padding=1)
    edges_y = F.conv2d(target_float, kernel_y, padding=1)
    
    # 计算梯度幅值
    # 使用 abs() 或 sqrt(edges_x**2 + edges_y**2)
    # 为了简单和与原始代码意图接近（尽管原始代码的Laplacian核不同），我们这里用绝对值和
    # edges = torch.abs(edges_x) + torch.abs(edges_y)
    # 或者更标准的梯度幅值
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    # 将边缘强度归一化到0-1（可选，但有助于阈值选择）
    # batch_size = edges.shape[0]
    # for i in range(batch_size):
    #     current_edge = edges[i]
    #     if current_edge.max() > 0:
    #         edges[i] = (current_edge - current_edge.min()) / (current_edge.max() - current_edge.min())

    # 根据阈值二值化得到边缘掩码
    edge_mask = (edges > threshold).float() 
    return edge_mask