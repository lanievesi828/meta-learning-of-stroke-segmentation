# 文件: src/pretrain.py

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR # 使用一个简单的调度器# 用于学习率衰减
from torch.optim.lr_scheduler import ReduceLROnPlateau # 用于学习率衰减
import os
import time
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import cv2 # 用于border_mode

# 导入你的模块
from src import config
from src.datasets import MedicalImageSegmentationDataset # 我们将复用这个基础加载器
from src.models import MAML_BiFPN_Transformer_UNet
#from src.models import OldArchMAML
# 损失函数也可以复用，但BCEWithLogitsLoss更标准
from torch.nn import BCEWithLogitsLoss 
from src.losses import MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss # 或者继续用你的组合损失
from src.utils import set_seed, dice_coefficient, iou_score
import albumentations as A

# --- 1. 为预训练创建专门的Dataset和DataLoader ---

class PretrainDataset(Dataset):
    """一个标准的数据集，接收图像和掩码路径列表。"""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        # 复用你的基础样本加载器，但不传递transform，因为我们在__getitem__中应用
        self.sample_loader = MedicalImageSegmentationDataset(transform=None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]
        
        # 加载原始的Numpy数据
        img_np = np.load(img_path)
        msk_np = np.load(msk_path)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=img_np, mask=msk_np)
            img_np = augmented['image']
            msk_np = augmented['mask']
        
        # 增加通道维度并转为Tensor
        img_np = np.expand_dims(img_np, axis=0)
        msk_np = np.expand_dims(msk_np, axis=0)

        image = torch.from_numpy(img_np.astype(np.float32))
        mask = torch.from_numpy(msk_np.astype(np.float32))
        #mask = (mask > 0.5).float() # 确保掩码是二值的

        return image, mask

def get_all_paths_from_json(json_path):
    """从任务JSON文件中提取所有唯一的图像和掩码路径。"""
    with open(json_path, 'r') as f:
        tasks = json.load(f)
    
    all_image_paths = []
    all_mask_paths = []
    for task_info in tasks.values():
        all_image_paths.extend(task_info['support_set_paths'])
        all_image_paths.extend(task_info['query_set_paths'])
        all_mask_paths.extend(task_info['support_set_mask_paths'])
        all_mask_paths.extend(task_info['query_set_mask_paths'])
        
    unique_paths = sorted(list(set(zip(all_image_paths, all_mask_paths))))
    images, masks = zip(*unique_paths)
    return list(images), list(masks)

# --- 2. 主预训练函数 ---
class EdgeEnhancedDiceLoss(nn.Module):
    def __init__(self, squared_denom=False, edge_weight=3.0):
        super(EdgeEnhancedDiceLoss, self).__init__()
        self.smooth = sys.float_info.epsilon
        self.squared_denom = squared_denom
        self.edge_weight = edge_weight
        
    def get_edge_mask(self, target):
        # Use Sobel operator to extract edges
        kernel = torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], 
                            dtype=torch.float32, device=target.device).view(1,1,3,3)
        edges = F.conv2d(target.float(), kernel, padding=1).abs()
        return (edges > 0).float()
        
    def forward(self, x, target):
        # Flatten inputs for dice calculation
        x_flat = x.view(-1)
        target_flat = target.view(-1)
        
        # Standard dice calculation
        intersection = (x_flat * target_flat).sum()
        numer = 2. * intersection + self.smooth
        factor = 2 if self.squared_denom else 1
        denom = x_flat.pow(factor).sum() + target_flat.pow(factor).sum() + self.smooth
        base_dice = numer / denom
        base_dice_loss = 1 - base_dice
        
        # Skip edge calculation if edge weight is 0
        if self.edge_weight <= 0:
            return base_dice_loss
        
        # For edge detection, we need to keep the 2D structure
        # No need to guess the shape, we can just use the original shapes
        batch_size = x.size(0)
        
        # Get edge mask using the target in its original form (before flattening)
        edge_mask = self.get_edge_mask(target)
        
        # Apply edge mask
        x_edges = x * edge_mask
        target_edges = target * edge_mask
        
        # Flatten for dice calculation
        x_edges_flat = x_edges.view(-1)
        target_edges_flat = target_edges.view(-1)
        
        # Edge dice calculation 
        edge_intersection = (x_edges_flat * target_edges_flat).sum()
        edge_numer = 2. * edge_intersection + self.smooth
        edge_denom = x_edges_flat.pow(factor).sum() + target_edges_flat.pow(factor).sum() + self.smooth
        edge_dice = edge_numer / edge_denom
        edge_dice_loss = 1 - edge_dice
        
        # Combine base dice loss and edge-enhanced dice loss
        total_loss = base_dice_loss + self.edge_weight * edge_dice_loss
        
        return total_loss

class BCEWithLogitsAndEdgeEnhancedDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.1, edge_weight=3.0, smooth=1.):
        super(BCEWithLogitsAndEdgeEnhancedDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = EdgeEnhancedDiceLoss(edge_weight=edge_weight)
        
    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        dice_loss = self.dice_loss(torch.sigmoid(inputs), targets)
        loss = self.bce_weight * bce_loss + (1. - self.bce_weight) * dice_loss
        return loss.mean()

def main_pretrain():
    set_seed(config.SEED)
    device = config.DEVICE
    
    # --- 初始化目录 ---
    pretrain_output_dir = os.path.join(config.BASE_OUTPUT_DIR, "pretrain_best_model")
    os.makedirs(pretrain_output_dir, exist_ok=True)
    print(f"预训练输出将保存到: {pretrain_output_dir}")

    # --- 数据增强管道 (不变) ---
    train_transforms = None
    val_transforms = None

    # --- 创建数据加载器 (不变) ---
    train_img_paths, train_mask_paths = get_all_paths_from_json(config.TRAIN_TASKS_JSON_PATH)
    val_img_paths, val_mask_paths = get_all_paths_from_json(config.VAL_TASKS_JSON_PATH)
    train_dataset = PretrainDataset(train_img_paths, train_mask_paths, transform=train_transforms)
    val_dataset = PretrainDataset(val_img_paths, val_mask_paths, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"预训练数据: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")

    # --- 初始化模型和损失函数 (不变) ---
    model = MAML_BiFPN_Transformer_UNet(
        in_channels=config.MODEL_IN_CHANNELS, 
        classes=config.MODEL_CLASSES,
        bifpn_channels=config.BIFPN_CHANNELS # 确保config中有这个值，比如64
    ).to(device)
    
    criterion = BCEWithLogitsAndEdgeEnhancedDiceLoss(bce_weight=0.1) # 沿用旧脚本的bce_weight
    
    # --- 优化器和学习率调度器 (核心修改) ---
    # 我们可以为预训练设置一个较高的最大学习率
    pretrain_lr = 1e-3 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_pretrain_epochs = 100
     # 预训练轮数，可以根据需要调整

    # *** 核心修改 1: 实例化 OneCycleLR ***
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.2,
        patience=3, # 使用旧脚本的 "急躁" 设置
        verbose=True
    )

    # --- 训练循环 ---
    best_val_dice = -1.0
    
    print(f"开始标准监督学习预训练，共 {num_pretrain_epochs} 个 epochs...")
    
    for epoch in range(num_pretrain_epochs):
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_pretrain_epochs}")
        for images, masks in train_progress:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, masks)
            loss.backward(); optimizer.step(); train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        val_loss, val_dice, val_iou, dice_samples, iou_samples = 0.0, 0.0, 0.0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks); val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                for i in range(images.size(0)):
                    pred_mask, gt_mask = preds[i], masks[i]
                    if gt_mask.sum() > 0 or pred_mask.sum() > 0:
                        val_dice += dice_coefficient(pred_mask, gt_mask).item(); dice_samples += 1
                        val_iou += iou_score(pred_mask > 0.5, gt_mask > 0.5).item(); iou_samples += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / dice_samples if dice_samples > 0 else 0.0
        avg_val_iou = val_iou / iou_samples if iou_samples > 0 else 0.0
        
        scheduler.step(avg_val_loss)
        
        # *** 核心修改 3: 移除 epoch 后的 scheduler.step() 调用 ***
        
        print(f"\nEpoch {epoch+1} 完成 | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证Dice: {avg_val_dice:.4f}")

        # 保存最佳模型 (不变)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_path = os.path.join(pretrain_output_dir, 'best_pretrained_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  *** 新的最佳验证Dice: {best_val_dice:.4f}。模型已保存到 {best_model_path} ***")
            
    print(f"\n预训练完成！最佳验证Dice为: {best_val_dice:.4f}")
if __name__ == '__main__':
    # 确保你的 MedicalImageSegmentationDataset 存在于 src/datasets.py 中
    try:
        from src.datasets import MedicalImageSegmentationDataset
    except ImportError:
        print("错误: 无法从 src.datasets 导入 MedicalImageSegmentationDataset。请确保文件和类存在。")
        exit()
        
    main_pretrain()