# 文件: src/visualize_predictions.py
# 版本: 最终出版级布局精调版 v4 (优化垂直间距，布局更紧凑)

import torch
import torch.optim as optim
import os
import copy
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
from skimage.measure import find_contours
from matplotlib.lines import Line2D

# --- 导入您的项目模块 ---
from src import config
from src.datasets import MetaTaskDataset
from src.models import MAML_BiFPN_Transformer_UNet
from src.losses import MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss
from src.utils import set_seed

# --- 1. 配置区域 ---
MODEL_PATHS = {
    "SSL_best_model": "/root/autodl-tmp/maml_data_functional_tasks_baseline_train/checkpoints/best_model.pth",
    "E2E_MAML_best_model": "/root/autodl-tmp/maml_data_functional_tasks_e2emaml/checkpoints_maml/best_maml_model.pth",
    "Pretrain_MAML_best_model": "/root/autodl-tmp/maml_data_functional_tasks_finetune_maml/checkpoints_maml/best_maml_model.pth"
}
VAL_TASKS_JSON_PATH = config.VAL_TASKS_JSON_PATH
OUTPUT_IMAGE_PATH_PDF = "/root/maml_stroke_segmentation/Fig6_Qualitative_Comparison.pdf"
OUTPUT_IMAGE_PATH_PNG = "/root/maml_stroke_segmentation/Fig6_Qualitative_Comparison.png"
FINETUNE_LR = 1e-3
FINETUNE_STEPS = config.NUM_INNER_STEPS

# --- 定义出版级的颜色和线条样式 ---
COLOR_GT = '#0072B2'
COLOR_PRED = '#D55E00'
LINEWIDTH = 1.0

# --- 辅助函数 (保持不变) ---
# ... (您的 get_model_prediction, get_ssl_finetune_prediction, plot_contours 函数放在这里) ...
def get_model_prediction(model, support_x, support_y, query_x, device):
    # ...
    temp_model = copy.deepcopy(model)
    temp_model.train()
    fast_weights = OrderedDict(temp_model.named_parameters())
    criterion = MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss().to(device)
    for _ in range(FINETUNE_STEPS):
        support_logits = temp_model(support_x, params=fast_weights)
        inner_loss = criterion(support_logits, support_y, reduction='mean')
        grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=False, allow_unused=True)
        fast_weights = OrderedDict((name, param - config.INNER_LR * grad if grad is not None else param) for ((name, param), grad) in zip(fast_weights.items(), grads))
    temp_model.eval()
    with torch.no_grad():
        query_logits = temp_model(query_x, params=fast_weights)
    return (torch.sigmoid(query_logits) > 0.5).float()

def get_ssl_finetune_prediction(model, support_x, support_y, query_x, device):
    # ...
    temp_model = copy.deepcopy(model)
    temp_model.train()
    optimizer = optim.Adam(temp_model.parameters(), lr=FINETUNE_LR)
    criterion = MAML_BCEWithLogitsAndEdgeEnhancedDiceLoss().to(device)
    for _ in range(FINETUNE_STEPS):
        optimizer.zero_grad()
        support_logits = temp_model(support_x)
        loss = criterion(support_logits, support_y, reduction='mean')
        loss.backward()
        optimizer.step()
    temp_model.eval()
    with torch.no_grad():
        query_logits = temp_model(query_x)
    return (torch.sigmoid(query_logits) > 0.5).float()

def plot_contours(ax, image, mask, color, linewidth):
    ax.imshow(image, cmap='gray')
    contours = find_contours(mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)

# --- 为Matplotlib设置全局参数 ---
def setup_publication_quality_matplotlib():
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['figure.titlesize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def main():
    set_seed(random.randint(0, 10000))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 加载模型 (保持不变) ---
    # ... (您的加载代码) ...
    print("加载所有预训练模型...")
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"警告: 找不到模型文件 {path}，跳过该模型。")
            continue
        model = MAML_BiFPN_Transformer_UNet(in_channels=1, classes=1).to(device)
        state_dict = torch.load(path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        models[name] = model
        print(f"成功加载模型: {name}")
    if len(models) < 3:
        print("错误：未能加载所有必需的模型，无法进行对比。")
        return

    # --- 2. 挑选任务 (保持不变) ---
    # ... (您的挑选任务代码) ...
    print("随机挑选一个验证任务...")
    val_dataset = MetaTaskDataset(tasks_json_path=VAL_TASKS_JSON_PATH)
    task_idx = random.randint(0, len(val_dataset) - 1)
    task_data = val_dataset[task_idx]
    support_x = task_data['support_images'].to(device)
    support_y = task_data['support_masks'].to(device)
    query_x = task_data['query_images'].to(device)
    query_y = task_data['query_masks'].to(device)
    query_slice_idx = random.randint(0, query_y.size(0) - 1)
    image_to_show = query_x[query_slice_idx].cpu().squeeze().numpy()
    gt_mask_to_show = query_y[query_slice_idx].cpu().squeeze().numpy()
    print(f"已随机选择任务 #{task_idx}，并从中选择查询切片 #{query_slice_idx} 进行可视化。")

    # --- 3. 生成预测 (保持不变) ---
    # ... (您的生成预测代码) ...
    print("为每个模型生成预测...")
    predictions = {}
    plot_order = ["SSL+Finetune", "E2E-MAML", "Pretrain-MAML"]
    model_name_map = {
        "SSL+Finetune": "SSL_best_model",
        "E2E-MAML": "E2E_MAML_best_model",
        "Pretrain-MAML": "Pretrain_MAML_best_model"
    }
    
    for method_name in plot_order:
        model_key = model_name_map[method_name]
        if model_key in models:
            if method_name == "SSL+Finetune":
                predictions[method_name] = get_ssl_finetune_prediction(models[model_key], support_x, support_y, query_x, device)[query_slice_idx].cpu().squeeze().numpy()
            else:
                predictions[method_name] = get_model_prediction(models[model_key], support_x, support_y, query_x, device)[query_slice_idx].cpu().squeeze().numpy()

    # --- 4. 绘图 (核心修改部分) ---
    print("开始绘制出版级对比图...")
    setup_publication_quality_matplotlib()

    num_methods = len(predictions)
    num_plots = 2 + num_methods
    
    # 【最终修改】减小figsize的高度，让整体更紧凑
    fig, axes = plt.subplots(1, num_plots, figsize=(8, 2.8))

    fig.suptitle('Qualitative Comparison of Few-shot Segmentation Performance', fontweight='bold')

    titles = ["(a) Input Image", "(b) Ground Truth", "(c) SSL+Finetune", "(d) E2E-MAML", "(e) Pretrain-MAML"]
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(image_to_show, cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

        if i == 1: # Ground Truth
            plot_contours(ax, image_to_show, gt_mask_to_show, color=COLOR_GT, linewidth=LINEWIDTH)
        elif i > 1: # Predictions
            method_name = plot_order[i-2]
            if method_name in predictions:
                plot_contours(ax, image_to_show, gt_mask_to_show, color=COLOR_GT, linewidth=LINEWIDTH)
                plot_contours(ax, image_to_show, predictions[method_name], color=COLOR_PRED, linewidth=LINEWIDTH)
    
    # --- 创建图例 ---
    legend_elements = [Line2D([0], [0], color=COLOR_GT, lw=2, label='Ground Truth'),
                       Line2D([0], [0], color=COLOR_PRED, lw=2, label='Prediction')]
    
    # 【最终修改】微调图例的垂直位置，使其更靠近子图
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    
    # 【最终修改】最终的布局参数，以实现紧凑效果
    fig.subplots_adjust(
        left=0.02, 
        right=0.98, 
        bottom=0.15, # bottom值稍大，因为图例本身占一定高度
        top=0.85,    # top值减小，让总标题更靠近子图
        wspace=0.25  # 减小水平间距
    )

    # --- 5. 保存 ---
    plt.savefig(OUTPUT_IMAGE_PATH_PDF, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(OUTPUT_IMAGE_PATH_PNG, dpi=600, bbox_inches='tight', pad_inches=0.05)
    
    plt.close(fig)
    print(f"可视化对比图已成功保存到: {OUTPUT_IMAGE_PATH_PDF} 和 {OUTPUT_IMAGE_PATH_PNG}")

if __name__ == '__main__':
    output_dir = os.path.dirname(OUTPUT_IMAGE_PATH_PDF)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main()