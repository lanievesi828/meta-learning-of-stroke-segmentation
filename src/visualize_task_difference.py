# 文件: visualize_tasks_for_publication.py
# 版本: 终极手动精确布局版 v2 (解决组标题对齐问题)

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- 1. 全局配置区域 ---
CONFIG = {
    "conventional_task_json": "/root/autodl-tmp/maml_data_heterogeneous_3s5q/train_meta_tasks.json",
    "functional_task_json": "/root/autodl-tmp/maml_data_functional_tasks/train_functional_meta_tasks.json",
    "output_image_basename": "/root/maml_stroke_segmentation/Fig4_Task_Comparison",
    "num_samples_to_show": 3,
    "specific_functional_category": "Cortical"
}

def setup_publication_quality_matplotlib():
    """为Matplotlib设置符合期刊出版质量的全局参数"""
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['font.size'] = 9 # 微调基础字号
    matplotlib.rcParams['axes.titlesize'] = 11 # 组标题字号
    matplotlib.rcParams['axes.labelsize'] = 10  # 行标签字号
    matplotlib.rcParams['figure.titlesize'] = 14 # 总标题
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def load_images_from_paths(img_paths, msk_paths):
    """从路径列表加载图像和掩码"""
    images = [np.load(p) for p in img_paths]
    masks = [np.load(p) for p in msk_paths]
    return images, masks

def plot_task_images(fig, gs, images, masks, start_col, is_left_group=False):
    """只绘制图像和掩码，不处理标题"""
    num_samples = len(images)
    for i in range(num_samples):
        # 上排画图像
        img_ax = fig.add_subplot(gs[0, start_col + i])
        img_ax.imshow(images[i], cmap='gray')
        img_ax.axis('off')
        if is_left_group and i == 0:
            img_ax.set_ylabel("Image", rotation=0, ha='right', va='center', labelpad=25)
        
        # 下排画掩码
        mask_ax = fig.add_subplot(gs[1, start_col + i])
        mask_ax.imshow(masks[i], cmap='gray')
        mask_ax.axis('off')
        if is_left_group and i == 0:
            mask_ax.set_ylabel("Mask", rotation=0, ha='right', va='center', labelpad=25)

def main():
    """主函数，生成出版级任务对比图"""
    print("开始生成任务定义对比图 (出版级)...")
    
    setup_publication_quality_matplotlib()

    # --- 加载和挑选任务数据 ---
    try:
        with open(CONFIG["conventional_task_json"], 'r') as f:
            conventional_tasks = json.load(f)
        with open(CONFIG["functional_task_json"], 'r') as f:
            functional_tasks = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 找不到任务JSON文件. {e}"); return

    # ... (您的数据加载逻辑保持不变) ...
    conv_task_name = random.choice(list(conventional_tasks.keys()))
    conv_task_def = conventional_tasks[conv_task_name]
    conv_imgs, conv_msks = load_images_from_paths(
        conv_task_def['support_set_paths'][:CONFIG["num_samples_to_show"]],
        conv_task_def['support_set_mask_paths'][:CONFIG["num_samples_to_show"]]
    )
    print(f"已选择传统任务: {conv_task_name}")
    
    category = CONFIG["specific_functional_category"]
    func_tasks_of_category = {n: d for n, d in functional_tasks.items() if d.get("task_category") == category}
    if not func_tasks_of_category:
        print(f"错误: 找不到类别为 '{category}' 的任务。"); return
    func_task_name = random.choice(list(func_tasks_of_category.keys()))
    func_task_def = functional_tasks[func_task_name]
    func_imgs, func_msks = load_images_from_paths(
        func_task_def['support_set_paths'][:CONFIG["num_samples_to_show"]],
        func_task_def['support_set_mask_paths'][:CONFIG["num_samples_to_show"]]
    )
    print(f"已选择功能性任务: {func_task_name} (类别: {category})")
    
    # --- 绘图 ---
    num_cols_per_group = CONFIG["num_samples_to_show"]
    num_total_cols = num_cols_per_group * 2 + 1 # 3 + 1 + 3 = 7
    
    fig = plt.figure(figsize=(6.7, 3.0)) 

    fig.suptitle('Visual comparison of support sets under different task paradigms', 
                 fontweight='bold', y=0.96)

    gs = fig.add_gridspec(
        nrows=2, 
        ncols=num_total_cols, 
        width_ratios=[1] * num_cols_per_group + [0.2] + [1] * num_cols_per_group,
        wspace=0.1, hspace=0.1,
        left=0.1, right=0.9, bottom=0.05, top=0.8
    )
    
    # --- 绘制图像 ---
    plot_task_images(fig, gs, conv_imgs, conv_msks, start_col=0, is_left_group=True)
    plot_task_images(fig, gs, func_imgs, func_msks, start_col=num_cols_per_group + 1)
    
    # 【核心修改】使用fig.text()进行绝对坐标定位来放置组标题
    # (x, y) 坐标是相对于整个figure的，(0,0)是左下角, (1,1)是右上角
    
    # 计算左侧组的中心x坐标
    left_group_center_x = (gs[0, 0].get_position(fig).x0 + gs[0, num_cols_per_group - 1].get_position(fig).x1) / 2
    # 计算右侧组的中心x坐标
    right_group_start_col = num_cols_per_group + 1
    right_group_end_col = num_total_cols - 1
    right_group_center_x = (gs[0, right_group_start_col].get_position(fig).x0 + gs[0, right_group_end_col].get_position(fig).x1) / 2
    
    # 定义标题的y坐标 (在子图区域的顶部)
    title_y = gs[0, 0].get_position(fig).y1 + 0.05 # 在子图顶部再往上一点
    
    # 放置左侧标题
    fig.text(left_group_center_x, title_y, "(a) Conventional Task", 
             ha='center', va='center', fontweight='bold',
             fontsize=matplotlib.rcParams['axes.titlesize'])
             
    # 放置右侧标题
    fig.text(right_group_center_x, title_y, f"(b) Functional Task ({category})", 
             ha='center', va='center', fontweight='bold',
             fontsize=matplotlib.rcParams['axes.titlesize'])
        
    # --- 保存 ---
    basename = CONFIG["output_image_basename"]
    output_path_pdf = f"{basename}.pdf"
    output_path_png = f"{basename}.png"
    
    output_dir = os.path.dirname(basename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path_pdf)
    print(f"矢量图 (PDF) 已成功保存到: {output_path_pdf}")

    plt.savefig(output_path_png, dpi=600, transparent=True)
    print(f"高质量位图 (PNG) 已成功保存到: {output_path_png}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()