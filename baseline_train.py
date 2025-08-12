import json
import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 2D  U-Net Transformer Architecture

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
        
class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe
    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000**(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y
        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)
    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)
        
class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)
    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

# BiFPN Components

class WeightedFeatureFusion(nn.Module):
    """
    WeightedFeatureFusion: Fast normalized feature fusion
    """
    def __init__(self, in_channels, out_channels, num_inputs=2, epsilon=1e-4):
        super(WeightedFeatureFusion, self).__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, *xs):
        # Get normalized weights
        weights = F.relu(self.weights)
        norm_weights = weights / (torch.sum(weights) + self.epsilon)
        
        # Weighted feature fusion
        output = sum(norm_weights[i] * xs[i] for i in range(len(xs)))
        output = self.conv(output)
        return output

class BiFPNBlock(nn.Module):
    def __init__(self, channels):
        super(BiFPNBlock, self).__init__()
        # Top-down pathway
        self.td_fusion_1 = WeightedFeatureFusion(channels, channels)
        self.td_fusion_2 = WeightedFeatureFusion(channels, channels)
        self.td_fusion_3 = WeightedFeatureFusion(channels, channels)
        
        # Bottom-up pathway
        self.bu_fusion_1 = WeightedFeatureFusion(channels, channels)
        self.bu_fusion_2 = WeightedFeatureFusion(channels, channels)
        self.bu_fusion_3 = WeightedFeatureFusion(channels, channels)
        
        # Learnable upsampling layers (replacing simple bilinear upsampling)
        self.upsample_4_to_3 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample_3_to_2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample_2_to_1 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        
        # Learnable downsampling layers (replacing max pooling)
        self.downsample_1_to_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.downsample_2_to_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.downsample_3_to_4 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, p1, p2, p3, p4):
        # Top-down pathway (from high level to low level)
        # P4 remains unchanged initially
        p4_td = p4
        
        # P3 gets input from P4
        p3_td_input = self.upsample_4_to_3(p4_td)
        # Ensure the shapes match
        if p3_td_input.shape != p3.shape:
            p3_td_input = F.interpolate(p3_td_input, size=p3.shape[2:])
        p3_td = self.td_fusion_1(p3, p3_td_input)
        
        # P2 gets input from P3
        p2_td_input = self.upsample_3_to_2(p3_td)
        # Ensure the shapes match
        if p2_td_input.shape != p2.shape:
            p2_td_input = F.interpolate(p2_td_input, size=p2.shape[2:])
        p2_td = self.td_fusion_2(p2, p2_td_input)
        
        # P1 gets input from P2
        p1_td_input = self.upsample_2_to_1(p2_td)
        # Ensure the shapes match
        if p1_td_input.shape != p1.shape:
            p1_td_input = F.interpolate(p1_td_input, size=p1.shape[2:])
        p1_td = self.td_fusion_3(p1, p1_td_input)
        
        # Bottom-up pathway (from low level to high level)
        # P1_out is same as p1_td
        p1_out = p1_td
        
        # P2 gets input from P1
        p2_bu_input = self.downsample_1_to_2(p1_out)
        # Ensure the shapes match
        if p2_bu_input.shape != p2_td.shape:
            p2_bu_input = F.interpolate(p2_bu_input, size=p2_td.shape[2:])
        p2_out = self.bu_fusion_1(p2_td, p2_bu_input)
        
        # P3 gets input from P2
        p3_bu_input = self.downsample_2_to_3(p2_out)
        # Ensure the shapes match
        if p3_bu_input.shape != p3_td.shape:
            p3_bu_input = F.interpolate(p3_bu_input, size=p3_td.shape[2:])
        p3_out = self.bu_fusion_2(p3_td, p3_bu_input)
        
        # P4 gets input from P3
        p4_bu_input = self.downsample_3_to_4(p3_out)
        # Ensure the shapes match
        if p4_bu_input.shape != p4_td.shape:
            p4_bu_input = F.interpolate(p4_bu_input, size=p4_td.shape[2:])
        p4_out = self.bu_fusion_3(p4_td, p4_bu_input)
        
        return p1_out, p2_out, p3_out, p4_out

class ChannelAdjustment(nn.Module):
    """
    Adjust number of channels if needed 
    """
    def __init__(self, in_channels, out_channels):
        super(ChannelAdjustment, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BiFPN_Transformer_UNet(nn.Module):
    def __init__(self, in_channels=1, classes=1, bilinear=True):
        super(BiFPN_Transformer_UNet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Self-attention at bottleneck
        self.MHSA = MultiHeadSelfAttention(512)
        
        # Channel adjustments for BiFPN (if needed)
        self.adjust_p1 = ChannelAdjustment(64, 64)    # P1 (feature level 1) adjustment
        self.adjust_p2 = ChannelAdjustment(128, 64)   # P2 adjustment
        self.adjust_p3 = ChannelAdjustment(256, 64)   # P3 adjustment
        self.adjust_p4 = ChannelAdjustment(512, 64)   # P4 adjustment
        
        # BiFPN Block
        self.bifpn = BiFPNBlock(64)
        
        # Final processing for output
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.outc = OutConv(64, classes)
        
    def forward(self, x):
        # Encoder (down) path
        x1 = self.inc(x)         # 64 channels
        x2 = self.down1(x1)      # 128 channels
        x3 = self.down2(x2)      # 256 channels
        x4 = self.down3(x3)      # 512 channels
        
        # Apply self-attention at bottleneck
        x4 = self.MHSA(x4)
        
        # Adjust channels for BiFPN
        p1 = self.adjust_p1(x1)  # Adjusted to 64
        p2 = self.adjust_p2(x2)  # Adjusted to 64
        p3 = self.adjust_p3(x3)  # Adjusted to 64
        p4 = self.adjust_p4(x4)  # Adjusted to 64
        
        # Apply BiFPN (bidirectional feature fusion)
        p1_out, p2_out, p3_out, p4_out = self.bifpn(p1, p2, p3, p4)
        
        # Use p1_out (lowest level features) for the final segmentation
        # This replaces the traditional decoder path with BiFPN's feature fusion
        out = self.final_conv(p1_out)
        
        # Output layer
        logits = self.outc(out)
        
        return logits

##################################         LOSS FUNCTION      ########################################
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
    
def dice_coefficient(inputs, labels, smooth=1):
    inputs = inputs.view(-1)
    labels = labels.view(-1)
    intersection = (inputs * labels).sum()
    union = inputs.sum() + labels.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# IOU
def IoU(output, labels):
    smooth = 1.
    intersection = torch.logical_and(output, labels).sum()
    union = torch.logical_or(output, labels).sum()
    return (intersection + smooth) / (union + smooth)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Training function
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    
    for batch in train_loader:
        inputs, labels = batch['image'], batch['mask']
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        epoch_loss += loss.item()
        dice = dice_coefficient(torch.sigmoid(outputs), labels).item()
        iou = IoU(outputs > 0.5, labels > 0.5).item()
        
        epoch_dice += dice
        epoch_iou += iou
    
    # Calculate average metrics for the epoch
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    avg_iou = epoch_iou / len(train_loader)
    
    return avg_loss, avg_dice, avg_iou

# Validation function
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    val_iou = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['image'], batch['mask']
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            val_loss += loss.item()
            dice = dice_coefficient(torch.sigmoid(outputs), labels).item()
            iou = IoU(outputs > 0.5, labels > 0.5).item()
            
            val_dice += dice
            val_iou += iou
    
    # Calculate average metrics for validation
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    
    return avg_val_loss, avg_val_dice, avg_val_iou

# Save checkpoints
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")

class SupervisedDatasetFromJson(Dataset):
    """
    一个为标准监督学习设计的数据集，但从MAML的JSON任务文件中加载数据。
    """
    def __init__(self, json_path, transform=None):
        self.transform = transform
        self.image_paths, self.mask_paths = self._get_all_paths_from_json(json_path)
        
    def _get_all_paths_from_json(self, json_path):
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
            
        # 使用字典去重，保持对应关系
        path_dict = dict(zip(all_image_paths, all_mask_paths))
        unique_images = sorted(path_dict.keys())
        unique_masks = [path_dict[key] for key in unique_images]
        
        print(f"从 {os.path.basename(json_path)} 中加载了 {len(unique_images)} 个唯一样本。")
        return unique_images, unique_masks

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]
        
        img = np.load(img_path)
        msk = np.load(msk_path)
        
        img = np.expand_dims(img, axis=0)
        msk = np.expand_dims(msk, axis=0)
        
        # 你的旧脚本没有使用数据增强，所以我们这里也设为None
        if self.transform:
            # ...
            pass
        
        # 返回旧脚本期望的字典格式
        return {'image': torch.from_numpy(img).float(), 'mask': torch.from_numpy(msk).float()} 

# 文件: src/baseline_train.py
# 文件: src/baseline_train.py (或者任何你需要绘图的脚本)

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_history(metrics_history, output_dir):
    """
    根据记录的指标历史，绘制并保存训练曲线图。
    Args:
        metrics_history (list of dict): 一个列表，每个元素是包含该epoch指标的字典。
        output_dir (str): 保存生成图表的目录路径。
    """
    if not metrics_history:
        print("警告: 训练历史记录为空，无法绘制图表。")
        return

    # 将历史记录列表转换为 pandas DataFrame，方便操作
    metrics_df = pd.DataFrame(metrics_history)
    
    # 确保epoch列是整数类型，用于正确的x轴标签
    if 'epoch' in metrics_df.columns:
        metrics_df['epoch'] = metrics_df['epoch'].astype(int)
    else:
        # 如果没有epoch列，就创建一个
        metrics_df['epoch'] = range(1, len(metrics_df) + 1)

    # 创建一个 2x2 的图表布局
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Standard Supervised Training History', fontsize=20, y=0.95)

    # --- 1. 损失曲线 (Loss Curve) ---
    axs[0, 0].plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-o', label='Train Loss', markersize=4, alpha=0.8)
    axs[0, 0].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-o', label='Validation Loss', markersize=4, alpha=0.8)
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].set_title('Loss vs. Epochs', fontsize=14)
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # --- 2. Dice 分数曲线 (Dice Score Curve) ---
    axs[0, 1].plot(metrics_df['epoch'], metrics_df['val_dice'], 'g-o', label='Validation Dice', markersize=4)
    
    # 找到并标记最佳Dice分数的点
    if not metrics_df['val_dice'].empty:
        best_dice_row = metrics_df.loc[metrics_df['val_dice'].idxmax()]
        best_epoch = int(best_dice_row['epoch'])
        best_dice = best_dice_row['val_dice']
        axs[0, 1].axhline(y=best_dice, color='gold', linestyle='--', 
                          label=f'Best Dice: {best_dice:.4f} at Epoch {best_epoch}')
        # 在最高点做一个标记
        axs[0, 1].plot(best_epoch, best_dice, 'y*', markersize=15, label=f'Best Point')
    
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Dice Score', fontsize=12)
    axs[0, 1].set_title('Validation Dice vs. Epochs', fontsize=14)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # --- 3. IoU 分数曲线 (IoU Score Curve) ---
    axs[1, 0].plot(metrics_df['epoch'], metrics_df['val_iou'], 'm-o', label='Validation IoU', markersize=4)
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('IoU Score', fontsize=12)
    axs[1, 0].set_title('Validation IoU vs. Epochs', fontsize=14)
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # --- 4. 学习率曲线 (Learning Rate Curve) ---
    axs[1, 1].plot(metrics_df['epoch'], metrics_df['lr'], 'c-o', label='Learning Rate', markersize=4)
    axs[1, 1].set_xlabel('Epoch', fontsize=12)
    axs[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axs[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axs[1, 1].set_yscale('log') # 使用对数坐标轴，能更清晰地观察学习率的大范围变化
    axs[1, 1].legend()
    axs[1, 1].grid(True, which="both", linestyle='--', alpha=0.6)

    # 调整整体布局以避免重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # 保存图像
    plot_save_path = os.path.join(output_dir, 'training_history_plot.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close() # 关闭图像，释放内存
    
    print(f"训练历史图表已成功保存到: {plot_save_path}")
def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # --- 1. 核心修改：定义新的数据和输出目录 ---
    
    # 数据源是你新预处理流程的输出
    # 请确保这个路径是正确的！
    data_root = "/root/autodl-tmp/best_images_multicenter_nobg_alldices_3s5q" 
    
    # 为这个基线实验创建一个新的、独立的输出目录
    output_dir = "/root/autodl-tmp/baseline_output_final_comparison"
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. 核心修改：使用新的Dataset加载JSON文件 ---
    
    train_json_path = os.path.join(data_root, "train_meta_tasks.json")
    val_json_path = os.path.join(data_root, "val_meta_tasks.json")
    
    # 确保JSON文件存在
    if not os.path.exists(train_json_path) or not os.path.exists(val_json_path):
        print(f"错误: 在 {data_root} 中找不到 train_meta_tasks.json 或 val_meta_tasks.json")
        print("请确保你已经运行了新的预处理脚本，并正确设置了'data_root'路径。")
        return
        
    print("正在从JSON文件加载数据集...")
    # 假设你的旧脚本没有使用数据增强
    train_dataset = SupervisedDatasetFromJson(train_json_path, transform=None)
    val_dataset = SupervisedDatasetFromJson(val_json_path, transform=None)
    
    # DataLoader, Model, Criterion, Optimizer, Scheduler 的创建与旧脚本完全一致
    batch_size = 16  # 与旧脚本保持一致
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # --- 3. 初始化模型、损失、优化器 (完全沿用旧脚本的设置) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiFPN_Transformer_UNet(in_channels=1, classes=1).to(device)
    
    # 多GPU支持 (如果需要的话)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU!")
        model = DataParallel(model, device_ids=[0, 1])
    
    criterion = BCEWithLogitsAndEdgeEnhancedDiceLoss(bce_weight=0.1) # 沿用旧脚本的bce_weight
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # 沿用旧脚本的优化器和参数
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True) # 沿用旧脚本的调度器
    
    # --- 4. 训练循环 (与旧脚本完全一致) ---
    num_epochs = 100
    best_val_dice = 0.0
    metrics_history = []
    start_epoch = 0 # 始终从头开始
    
    print("开始标准监督学习训练 (使用旧模型架构，新数据划分)...")
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        # 调用与旧脚本相同的训练和验证函数
        train_loss, train_dice, train_iou = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_dataloader, criterion, device)
        
        # 调度器基于验证损失进行更新
        scheduler.step(val_loss)
        
        # --- 日志记录 ---
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_dice': train_dice, 'train_iou': train_iou,
            'val_loss': val_loss, 'val_dice': val_dice, 'val_iou': val_iou,
            'lr': optimizer.param_groups[0]['lr']
        }
        metrics_history.append(epoch_metrics)
        
        # --- 保存模型和检查点 ---
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            print(f"  *** 新的最佳验证Dice: {best_val_dice:.4f}. 保存最佳模型... ***")
            # 保存最佳模型权重
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
        
        # 每10个epoch保存一次完整的检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, epoch_metrics,
                os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # --- 打印日志 ---
        time_taken = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - Time: {time_taken:.2f}s')
        print(f'  Train -> Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'  Val   -> Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        print(f'  Best Val Dice: {best_val_dice:.4f}, Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 80)
        
        # --- 保存CSV历史记录 ---
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    print("训练完成。")
    
    # --- 最终绘图 ---
    plot_history(metrics_history, output_dir) # 假设你有一个plot_history函数


if __name__ == "__main__":
    main()