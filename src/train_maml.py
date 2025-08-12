import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# 【修正】OneCycleLR 是一个很好的选择，保持它
from torch.optim.lr_scheduler import OneCycleLR
from src.datasets import MetaTaskDataset, FunctionalTaskBatchSampler
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys # 引入sys以处理路径问题

# --- 【新增】确保能找到src模块 ---
# 这是一个好习惯，特别是当你的脚本不在项目根目录时
# 如果 train_maml.py 在根目录，这行可以省略
# 如果它在 src/ 下，也可以省略
# 如果它在 scripts/ 下，就需要这行
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.datasets import MetaTaskDataset
from src.models import MAML_BiFPN_Transformer_UNet
from src.losses import MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss
from src.maml_trainer import MAMLTrainer
from src.utils import set_seed, dice_coefficient, iou_score

# 你的 save_validation_visualizations 和 validate_meta_model 函数定义保持不变
# ... (将你的 save_validation_visualizations 和 validate_meta_model 函数完整复制到这里) ...
def save_validation_visualizations(support_images, support_masks, 
                                   query_images, query_masks, 
                                   query_pred_probs, query_pred_binary, 
                                   task_batch_index, task_index_in_batch, epoch, 
                                   output_dir, num_samples_to_show=2):
    """
    保存验证过程中的可视化图像。
    Args:
        support_images, support_masks: 当前任务的支持集图像和掩码 (K_SHOT, C, H, W)
        query_images, query_masks: 当前任务的查询集图像和掩码 (K_QUERY, C, H, W)
        query_pred_probs: 模型对查询集的预测概率图 (K_QUERY, C, H, W)
        query_pred_binary: 模型对查询集的二值化预测掩码 (K_QUERY, C, H, W)
        task_batch_index: 当前是第几个验证批次。
        task_index_in_batch: 当前是该批次中的第几个任务。
        epoch: 当前的元学习 epoch。
        output_dir: 保存图像的目录。
        num_samples_to_show: 每个集合（支持/查询）显示多少个样本。
    """
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    # --- 1. 可视化支持集 ---
    num_support_to_show = min(num_samples_to_show, support_images.size(0))
    if num_support_to_show > 0:
        fig_s, axes_s = plt.subplots(num_support_to_show, 2, figsize=(8, num_support_to_show * 4))
        if num_support_to_show == 1: # plt.subplots 返回的 axes 形状会不同
            axes_s = np.array([axes_s]) 
        fig_s.suptitle(f'Epoch {epoch+1} - Val Task (Batch {task_batch_index}, Task {task_index_in_batch}) - Support Set')
        for i in range(num_support_to_show):
            img_s = support_images[i].cpu().squeeze().numpy() # (H, W)
            msk_s = support_masks[i].cpu().squeeze().numpy()  # (H, W)
            
            axes_s[i, 0].imshow(img_s, cmap='gray')
            axes_s[i, 0].set_title(f'Support Image {i+1}')
            axes_s[i, 0].axis('off')
            
            axes_s[i, 1].imshow(msk_s, cmap='gray')
            axes_s[i, 1].set_title(f'Support Mask {i+1}')
            axes_s[i, 1].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应标题
        fig_s.savefig(os.path.join(output_dir, f'epoch{epoch+1}_valbatch{task_batch_index}_task{task_index_in_batch}_support.png'))
        plt.close(fig_s)

    # --- 2. 可视化查询集 (图像, 真值掩码, 预测概率, 预测二值掩码) ---
    num_query_to_show = min(num_samples_to_show, query_images.size(0))
    if num_query_to_show > 0:
        fig_q, axes_q = plt.subplots(num_query_to_show, 4, figsize=(16, num_query_to_show * 4))
        if num_query_to_show == 1: # plt.subplots 返回的 axes 形状会不同
            axes_q = np.array([axes_q])
        fig_q.suptitle(f'Epoch {epoch+1} - Val Task (Batch {task_batch_index}, Task {task_index_in_batch}) - Query Set')
        for i in range(num_query_to_show):
            img_q = query_images[i].cpu().squeeze().numpy()
            msk_q_gt = query_masks[i].cpu().squeeze().numpy()
            pred_prob_q = query_pred_probs[i].cpu().squeeze().numpy()
            pred_bin_q = query_pred_binary[i].cpu().squeeze().numpy()

            axes_q[i, 0].imshow(img_q, cmap='gray')
            axes_q[i, 0].set_title(f'Query Image {i+1}')
            axes_q[i, 0].axis('off')

            axes_q[i, 1].imshow(msk_q_gt, cmap='gray')
            axes_q[i, 1].set_title(f'GT Mask {i+1}')
            axes_q[i, 1].axis('off')

            axes_q[i, 2].imshow(pred_prob_q, cmap='viridis') # 用 viridis colormap 看概率图
            axes_q[i, 2].set_title(f'Pred Prob {i+1}')
            axes_q[i, 2].axis('off')

            axes_q[i, 3].imshow(pred_bin_q, cmap='gray')
            axes_q[i, 3].set_title(f'Pred Mask {i+1}')
            axes_q[i, 3].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_q.savefig(os.path.join(output_dir, f'epoch{epoch+1}_valbatch{task_batch_index}_task{task_index_in_batch}_query.png'))
        plt.close(fig_q)
def validate_meta_model(maml_trainer_instance, val_loader, epoch_num,
                        visualization_output_dir=None, visualize_every_n_batches=5):
    """
    在验证集上评估元模型的性能。
    这是整合了所有修正的最终版本。
    """
    maml_trainer_instance.model.eval()

    # --- 初始化所有累加器和计数器 ---
    total_val_query_loss = 0.0
    total_val_dice = 0.0
    dice_samples_count = 0  # 只计算有意义的样本数
    total_val_iou = 0.0
    iou_samples_count = 0   # IoU 的独立计数器
    num_val_tasks_processed = 0

    with torch.no_grad():
        for batch_idx, batched_val_task_data in enumerate(val_loader):
            all_val_support_images = batched_val_task_data['support_images'].to(config.DEVICE)
            all_val_support_masks = batched_val_task_data['support_masks'].to(config.DEVICE)
            all_val_query_images = batched_val_task_data['query_images'].to(config.DEVICE)
            all_val_query_masks = batched_val_task_data['query_masks'].to(config.DEVICE)

            num_tasks_in_current_batch = all_val_support_images.size(0)

            for task_idx_in_batch in range(num_tasks_in_current_batch):
                support_images = all_val_support_images[task_idx_in_batch]
                support_masks = all_val_support_masks[task_idx_in_batch]
                query_images = all_val_query_images[task_idx_in_batch]
                query_masks = all_val_query_masks[task_idx_in_batch]

                # --- 验证时的内循环适应 ---
                fast_weights = OrderedDict(maml_trainer_instance.model.named_parameters())
                for _ in range(maml_trainer_instance.num_inner_steps):
                    with torch.enable_grad():  # 局部启用梯度计算
                        support_logits = maml_trainer_instance.model(support_images, params=fast_weights)
                        inner_loss = maml_trainer_instance.loss_fn(support_logits, support_masks, reduction='mean')
                        
                        # 计算梯度，但不为元更新创建图
                        grads = torch.autograd.grad(
                            inner_loss, 
                            fast_weights.values(), 
                            create_graph=False,      # 关键点: 验证时为 False
                            allow_unused=True        # 关键点: 允许未使用参数
                        )
                        
                        # 更新快速权重，处理 None 梯度
                        fast_weights = OrderedDict(
                            (name, param - maml_trainer_instance.inner_lr * grad if grad is not None else param)
                            for ((name, param), grad) in zip(fast_weights.items(), grads)
                        )

                # --- 使用适应后的权重进行评估 ---
                query_pred_logits = maml_trainer_instance.model(query_images, params=fast_weights)
                
                # 计算损失
                query_loss = maml_trainer_instance.loss_fn(query_pred_logits, query_masks, reduction='mean')
                total_val_query_loss += query_loss.item()
                
                # 准备用于计算指标的张量
                query_pred_probs = torch.sigmoid(query_pred_logits)
                query_pred_binary = (query_pred_probs > 0.5).float() # 二值化并转为 0.0/1.0
                
                # --- 逐样本计算指标，并只在有意义的样本上累加 ---
                num_query_samples = query_masks.shape[0]
                if num_query_samples > 0:
                    for q_idx in range(num_query_samples):
                        pred_mask = query_pred_binary[q_idx]
                        gt_mask = query_masks[q_idx]
                        
                        # 核心判断：只在预测或真值不为空时计算指标
                        if pred_mask.sum() > 0 or gt_mask.sum() > 0:
                            # 计算并累加 Dice
                            current_dice = dice_coefficient(pred_mask, gt_mask).item()
                            total_val_dice += current_dice
                            dice_samples_count += 1
                            
                            # 计算并累加 IoU
                            current_iou = iou_score(pred_mask, gt_mask).item()
                            total_val_iou += current_iou
                            iou_samples_count += 1
                
                num_val_tasks_processed += 1

                # --- 可视化 ---
                if visualization_output_dir and \
                   batch_idx % visualize_every_n_batches == 0 and \
                   task_idx_in_batch == 0:
                    print(f"  保存可视化图像: Epoch {epoch_num+1}, Val Batch {batch_idx}...")
                    save_validation_visualizations(
                        support_images, support_masks,
                        query_images, query_masks,
                        query_pred_probs, query_pred_binary, # 传入概率图和二值图
                        batch_idx, task_idx_in_batch, epoch_num,
                        visualization_output_dir
                    )
    
    # --- 使用正确的计数器计算最终的平均指标 ---
    avg_val_loss = total_val_query_loss / num_val_tasks_processed if num_val_tasks_processed > 0 else 0
    avg_val_dice = total_val_dice / dice_samples_count if dice_samples_count > 0 else 0.0
    avg_val_iou = total_val_iou / iou_samples_count if iou_samples_count > 0 else 0.0

    return avg_val_loss, avg_val_dice, avg_val_iou
def main():
    """
    主训练函数，支持从检查点恢复或从头开始训练。
    【已更新以使用CenterBasedBatchSampler】
    """
    set_seed(config.SEED)

    # --- 1. 路径和配置 ---
    PRETRAINED_MODEL_PATH = config.PRETRAINED_MODEL_PATH
    RESUME_FROM_CHECKPOINT = '/root/autodl-tmp/maml_data_functional_tasks_e2emaml/checkpoints_maml/maml_checkpoint_epoch_60_1.pth' # or your maml checkpoint path

    vis_dir = os.path.join(config.RESULTS_DIR, "validation_visuals")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"验证集可视化图像将保存到: {vis_dir}")

    # --- 2. 数据加载【核心修改部分】---
    print("正在初始化数据集和数据加载器...")
    
    # a. 创建数据集实例 (这部分不变)
    train_meta_dataset = MetaTaskDataset(
        tasks_json_path=config.TRAIN_TASKS_JSON_PATH,
        transform=None # 假设训练时不用数据增强
    )
    val_meta_dataset = MetaTaskDataset(
        tasks_json_path=config.VAL_TASKS_JSON_PATH,
        transform=None
    )

    # b. 【新增】为训练集创建按中心批处理的采样器
    train_batch_sampler = FunctionalTaskBatchSampler(
        dataset=train_meta_dataset,
        batch_size=config.META_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_batch_sampler = FunctionalTaskBatchSampler(
        dataset=val_meta_dataset,
        batch_size=config.META_BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    # c. 【核心修改】在DataLoader中使用新的batch_sampler (变量名变了)
    train_meta_loader = DataLoader(
        train_meta_dataset,
        batch_sampler=train_batch_sampler, # <-- 使用新的采样器
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_meta_loader = DataLoader(
        val_meta_dataset,
        batch_sampler=val_batch_sampler, # <-- 使用新的采样器
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # 打印出的批次数现在会更准确
    print(f"训练任务加载器批次数: {len(train_meta_loader)}, 验证任务加载器批次数: {len(val_meta_loader)}")

    # --- 3. 初始化模型、损失、优化器 (这部分完全不变) ---
    print("正在初始化模型、损失函数和 MAML 训练器...")
    meta_model_instance = MAML_BiFPN_Transformer_UNet(
        in_channels=config.MODEL_IN_CHANNELS, 
        classes=config.MODEL_CLASSES,
        bifpn_channels=config.BIFPN_CHANNELS
    ).to(config.DEVICE)

    criterion = MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss(
        bce_weight=config.BCE_WEIGHT, 
        edge_weight=config.EDGE_WEIGHT,
    ).to(config.DEVICE)

    meta_optimizer = optim.AdamW(
        meta_model_instance.parameters(), 
        lr=config.OUTER_LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 调度器初始化 (保持不变)
    # 【重要】steps_per_epoch现在应该使用len(train_meta_loader)，它会从batch_sampler获取准确的批次数
    meta_scheduler = OneCycleLR(
        meta_optimizer,
        max_lr=config.OUTER_LR,
        steps_per_epoch=len(train_meta_loader),
        epochs=config.NUM_META_EPOCHS,
        anneal_strategy='cos'
    )
    
    maml_trainer = MAMLTrainer(
        model=meta_model_instance,
        loss_fn=criterion,
        inner_lr=config.INNER_LR,
        meta_optimizer=meta_optimizer,
        num_inner_steps=config.NUM_INNER_STEPS,
        device=config.DEVICE
    )
    
    # --- 4. 加载权重 (这部分完全不变) ---
    start_epoch = 0
    best_val_meta_dice = -1.0
    metrics_history = []

    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"正在从MAML检查点恢复: {RESUME_FROM_CHECKPOINT}")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=config.DEVICE)
        
        meta_model_instance.load_state_dict(checkpoint['model_state_dict'])
        meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        meta_scheduler.load_state_dict(checkpoint['meta_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_meta_dice = checkpoint['best_val_meta_dice']
        metrics_history = checkpoint.get('metrics_history', [])
        print(f"MAML训练已恢复。将从 Epoch {start_epoch + 1} 开始。")

    elif PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"未找到MAML检查点，正在从预训练模型加载权重: {PRETRAINED_MODEL_PATH}")
        pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
        meta_model_instance.load_state_dict(pretrained_dict)
        print("预训练权重加载成功。将从 Epoch 1 开始进行MAML训练。")
    
    else:
        print("警告：未找到MAML检查点或预训练模型。将从随机权重开始训练。")

    # --- 5. 训练循环 (这部分完全不变，因为逻辑已经解耦) ---
    print("开始 MAML 训练...")
    start_time_total_train = time.time()

    for epoch in range(start_epoch, config.NUM_META_EPOCHS):
        epoch_start_time = time.time()
        
        meta_model_instance.train()
        total_train_meta_loss_epoch = 0.0
        
        from tqdm import tqdm
        train_iterator = tqdm(train_meta_loader, desc=f"Epoch {epoch+1}/{config.NUM_META_EPOCHS}")

        for i, task_batch in enumerate(train_iterator):
            meta_loss_for_batch = maml_trainer.outer_loop_batch(task_batch)
            total_train_meta_loss_epoch += meta_loss_for_batch
            
            train_iterator.set_postfix(meta_loss=f"{meta_loss_for_batch:.4f}", lr=f"{meta_optimizer.param_groups[0]['lr']:.2e}")
            
            meta_scheduler.step()

        avg_train_meta_loss_epoch = total_train_meta_loss_epoch / len(train_meta_loader) if len(train_meta_loader) > 0 else 0

        avg_val_loss, avg_val_dice, avg_val_iou = validate_meta_model(
            maml_trainer, val_meta_loader, epoch,
            visualization_output_dir=vis_dir,
            visualize_every_n_batches=max(1, len(val_meta_loader) // 4)
        )
        
        epoch_duration = time.time() - epoch_start_time
        current_outer_lr = meta_optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{config.NUM_META_EPOCHS} | Time: {epoch_duration:.2f}s")
        print(f"  Train Meta-Loss: {avg_train_meta_loss_epoch:.4f}")
        print(f"  Val Meta-Loss:   {avg_val_loss:.4f} | Val Meta-Dice: {avg_val_dice:.4f} | Val Meta-IoU: {avg_val_iou:.4f}")
        print(f"  Current Outer LR: {current_outer_lr:.7f}")

        epoch_metrics = {
            'epoch': epoch + 1, 'train_meta_loss': avg_train_meta_loss_epoch,
            'val_meta_loss': avg_val_loss, 'val_meta_dice': avg_val_dice,
            'val_meta_iou': avg_val_iou, 'outer_lr': current_outer_lr
        }
        metrics_history.append(epoch_metrics)

        if avg_val_dice > best_val_meta_dice:
            best_val_meta_dice = avg_val_dice
            print(f"  *** 新的最佳验证集 Meta-Dice: {best_val_meta_dice:.4f}. 保存最佳MAML模型... ***")
            torch.save(meta_model_instance.state_dict(), 
                       os.path.join(config.CHECKPOINT_DIR, 'best_maml_model.pth'))
        
        if (epoch + 1) % 10 == 0 or epoch == config.NUM_META_EPOCHS - 1:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'maml_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1, 
                'model_state_dict': meta_model_instance.state_dict(),
                'meta_optimizer_state_dict': maml_trainer.meta_optimizer.state_dict(),
                'meta_scheduler_state_dict': meta_scheduler.state_dict(),
                'best_val_meta_dice': best_val_meta_dice,
                'metrics_history': metrics_history
            }, checkpoint_path)
            print(f"检查点已保存到 {checkpoint_path}")

        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(os.path.join(config.RESULTS_DIR, 'maml_training_history.csv'), index=False)

    total_training_time = time.time() - start_time_total_train
    print(f"\nMAML 训练完成。总耗时: {total_training_time/3600:.2f} 小时。")
    print(f"最佳验证 Meta-Dice: {best_val_meta_dice:.4f}")
    
    # --- 绘图 ---
    if metrics_history:
        print("正在绘制训练历史图表...")
        # ... (你的绘图代码保持不变，它是正确的) ...
        metrics_df = pd.DataFrame(metrics_history)
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('MAML Training and Validation History', fontsize=16)
        axs[0, 0].plot(metrics_df['epoch'], metrics_df['train_meta_loss'], 'b-o', label='Train Meta-Loss', markersize=4)
        axs[0, 0].plot(metrics_df['epoch'], metrics_df['val_meta_loss'], 'r-o', label='Val Meta-Loss', markersize=4)
        axs[0, 0].set_xlabel('Epoch'); axs[0, 0].set_ylabel('Loss'); axs[0, 0].set_title('Meta-Loss vs. Epochs'); axs[0, 0].legend(); axs[0, 0].grid(True, linestyle='--', alpha=0.6)
        axs[0, 1].plot(metrics_df['epoch'], metrics_df['val_meta_dice'], 'g-o', label='Val Meta-Dice', markersize=4)
        axs[0, 1].set_xlabel('Epoch'); axs[0, 1].set_ylabel('Dice Score'); axs[0, 1].set_title('Validation Meta-Dice vs. Epochs')
        if not metrics_df['val_meta_dice'].empty:
            best_dice_epoch = metrics_df.loc[metrics_df['val_meta_dice'].idxmax()]
            axs[0, 1].axhline(y=best_dice_epoch['val_meta_dice'], color='gold', linestyle='--', label=f'Best Dice: {best_dice_epoch["val_meta_dice"]:.4f} at Epoch {int(best_dice_epoch["epoch"])}')
        axs[0, 1].legend(); axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        axs[1, 0].plot(metrics_df['epoch'], metrics_df['val_meta_iou'], 'm-o', label='Val Meta-IoU', markersize=4)
        axs[1, 0].set_xlabel('Epoch'); axs[1, 0].set_ylabel('IoU Score'); axs[1, 0].set_title('Validation Meta-IoU vs. Epochs'); axs[1, 0].legend(); axs[1, 0].grid(True, linestyle='--', alpha=0.6)
        axs[1, 1].plot(metrics_df['epoch'], metrics_df['outer_lr'], 'c-o', label='Outer Learning Rate', markersize=4)
        axs[1, 1].set_xlabel('Epoch'); axs[1, 1].set_ylabel('Learning Rate'); axs[1, 1].set_title('Outer LR Schedule'); axs[1, 1].set_yscale('log'); axs[1, 1].legend(); axs[1, 1].grid(True, which="both", linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path = os.path.join(config.RESULTS_DIR, 'maml_training_plots.png')
        plt.savefig(plot_save_path); plt.close()
        print(f"训练历史图表已保存到: {plot_save_path}")

if __name__ == '__main__':
    if not os.path.exists(config.TRAIN_TASKS_JSON_PATH):
        print(f"错误: 训练任务JSON文件未找到: {config.TRAIN_TASKS_JSON_PATH}")
    else:
        main()