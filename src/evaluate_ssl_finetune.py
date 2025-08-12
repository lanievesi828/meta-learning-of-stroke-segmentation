# 文件: src/evaluate_ssl_finetune.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# --- 从你的项目中导入必要的模块 ---
# 确保你的Python环境能找到src目录
from src import config
from src.datasets import MetaTaskDataset
from src.models import MAML_BiFPN_Transformer_UNet
# 为了确保微调的损失函数与MAML内循环完全一致
from src.losses import MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss
from src.utils import set_seed, dice_coefficient, iou_score

# ==============================================================================
# 1. 配置区域 - 所有关键参数都在这里，方便调整
# ==============================================================================

# --- 指向你性能最佳的SSL预训练模型（Dice达到0.58的那个） ---
SSL_MODEL_PATH = "/root/autodl-tmp/maml_data_functional_tasks_baseline_train/checkpoints/best_model.pth"

# --- 微调（Fine-tuning）超参数 ---
# 这是在support set上进行小样本适应时使用的学习率
# 1e-3 是一个很好的起点，你也可以尝试 1e-2 或 1e-4
FINETUNE_LR = 1e-3

# 微调的梯度更新步数
# 为了公平比较，这个值必须与你MAML设置中的 NUM_INNER_STEPS 完全相同
FINETUNE_STEPS = config.NUM_INNER_STEPS  # 例如, 5

# ==============================================================================

def main():
    """
    主函数，用于运行 SSL+Finetune 评估。
    该脚本加载一个预训练的SSL模型，并在验证任务集上评估其小样本适应性能。
    """
    set_seed(config.SEED)
    device = config.DEVICE

    # --- 路径健全性检查 ---
    if not os.path.exists(SSL_MODEL_PATH):
        print(f"致命错误: SSL模型路径未找到: {SSL_MODEL_PATH}")
        print("请确保路径正确且模型文件存在。")
        return

    print("--- SSL模型小样本微调性能评估 ---")
    print(f"正在从以下路径加载SSL模型: {SSL_MODEL_PATH}")
    print(f"微调学习率: {FINETUNE_LR}, 微调步数: {FINETUNE_STEPS}")

    # --- 1. 加载预训练的SSL模型 ---
    # 首先实例化模型架构
    ssl_model = MAML_BiFPN_Transformer_UNet(
        in_channels=config.MODEL_IN_CHANNELS, 
        classes=config.MODEL_CLASSES,
        bifpn_channels=config.BIFPN_CHANNELS
    ).to(device)

    # 加载已保存的权重
    ssl_model.load_state_dict(torch.load(SSL_MODEL_PATH, map_location=device))
    print("SSL模型权重加载成功。")

    # --- 2. 设置用于微调的标准损失函数 ---
    # 我们使用与MAML兼容的损失函数，以确保微调过程
    # 在数学上与MAML的内循环完全等价
    finetune_criterion = MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss(
        bce_weight=config.BCE_WEIGHT, 
        edge_weight=config.EDGE_WEIGHT,
    ).to(device)

    # --- 3. 加载验证任务数据集 ---
    val_meta_dataset = MetaTaskDataset(
        tasks_json_path=config.VAL_TASKS_JSON_PATH,
        transform=None  # 这里可以添加必要的预处理转换
    )
    val_loader = DataLoader(
        val_meta_dataset,
        batch_size=config.META_BATCH_SIZE,  # 为了保持一致性，使用相同的批次大小
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"已加载 {len(val_meta_dataset)} 个验证任务。")

    # --- 4. 评估循环 ---
    # 初始化指标累加器，与你的MAML验证流程完全相同
    total_dice = 0.0
    dice_samples_count = 0
    total_iou = 0.0
    iou_samples_count = 0
    num_tasks_processed = 0

    # 主评估循环
    evaluation_iterator = tqdm(val_loader, desc="正在评估 SSL+Finetune")
    for task_batch in evaluation_iterator:
        
        # 解包一批任务数据
        support_images_batch = task_batch['support_images'].to(device)
        support_masks_batch = task_batch['support_masks'].to(device)
        query_images_batch = task_batch['query_images'].to(device)
        query_masks_batch = task_batch['query_masks'].to(device)

        # 遍历该批次中的每一个任务
        for i in range(support_images_batch.size(0)):
            
            # --- 关键步骤: 为每个任务创建一个模型的临时副本 ---
            # 这确保了每个任务都从完全相同的原始SSL模型开始微调。
            temp_model = copy.deepcopy(ssl_model)
            temp_model.train()  # 设置为训练模式以进行微调（例如，为了BatchNorm层）

            # 为这一个任务的微调创建一个专用的优化器
            finetune_optimizer = optim.Adam(temp_model.parameters(), lr=FINETUNE_LR)

            # 获取当前任务的数据
            support_x = support_images_batch[i]
            support_y = support_masks_batch[i]
            
            # --- 微调循环 (模拟MAML的内循环) ---
            for _ in range(FINETUNE_STEPS):
                finetune_optimizer.zero_grad()
                
                support_logits = temp_model(support_x)
                loss = finetune_criterion(support_logits, support_y, reduction='mean')
                
                loss.backward()
                finetune_optimizer.step()

            # --- 在Query集上进行评估 ---
            temp_model.eval()  # 设置为评估模式进行推理
            with torch.no_grad():
                query_x = query_images_batch[i]
                query_y = query_masks_batch[i]

                query_logits = temp_model(query_x)
                query_probs = torch.sigmoid(query_logits)
                query_preds_binary = (query_probs > 0.5).float()

                # 为query集中的每个样本计算指标
                for q_idx in range(query_y.shape[0]):
                    pred_mask = query_preds_binary[q_idx]
                    gt_mask = query_y[q_idx]

                    # 只在真值或预测不为空时计算，避免无意义的满分
                    if gt_mask.sum() > 0 or pred_mask.sum() > 0:
                        total_dice += dice_coefficient(pred_mask, gt_mask).item()
                        dice_samples_count += 1
                        total_iou += iou_score(pred_mask, gt_mask).item()
                        iou_samples_count += 1

            num_tasks_processed += 1
    
    # --- 5. 报告最终结果 ---
    avg_dice = total_dice / dice_samples_count if dice_samples_count > 0 else 0.0
    avg_iou = total_iou / iou_samples_count if iou_samples_count > 0 else 0.0
    
    print("\n--- 评估完成 ---")
    print(f"总共评估的任务数量: {num_tasks_processed}")
    print(f"评估方法: 预训练SSL模型 + {FINETUNE_STEPS}步微调")
    print(f"平均 Dice 分数: {avg_dice:.4f}")
    print(f"平均 IoU 分数:  {avg_iou:.4f}")

    # (可选) 将结果保存到文件
    results = {
        "method": "SSL+Finetune",
        "ssl_model_path": SSL_MODEL_PATH,
        "finetune_lr": FINETUNE_LR,
        "finetune_steps": FINETUNE_STEPS,
        "avg_dice": avg_dice,
        "avg_iou": avg_iou
    }
    # 确保结果目录存在
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, 'ssl_finetune_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"结果已保存至: {results_path}")

if __name__ == '__main__':
    # 添加一个简单的检查，确保配置文件中的路径有效
    if not os.path.exists(config.VAL_TASKS_JSON_PATH):
        print(f"致命错误: 验证任务JSON文件未找到: {config.VAL_TASKS_JSON_PATH}")
    else:
        main()