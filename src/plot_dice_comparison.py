# 文件: plot_for_publication.py
# 版本: 终极版 - 解决标题裁剪并优化注释

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

# --- 1. 全局配置区域 ---

# 【重要】请将这里的路径修改为你四个实验结果CSV文件的实际路径
FILE_PATHS = {
    "E2E-MAML (Conventional)": "/root/autodl-tmp/maml_data_heterogeneous_3s5q_maml/results_maml/maml_training_history.csv",
    "Pretrain-MAML (Conventional)": "/root/autodl-tmp/maml_stroke_output_transbu_with_bg_3s5q_newmodel/results_maml/maml_training_history.csv",
    "E2E-MAML (Functional)": "/root/autodl-tmp/maml_data_functional_tasks_e2emaml/results_maml/maml_training_history.csv",
    "Pretrain-MAML (Functional)": "/root/autodl-tmp/maml_data_functional_tasks_finetune_maml/results_maml/maml_training_history.csv"
}

# 输出配置
OUTPUT_CONFIG = {
    "save_path_pdf": "/root/autodl-tmp/JEI_MAML_Comparison_Final.pdf", # 首选矢量格式
    "save_path_png": "/root/autodl-tmp/JEI_MAML_Comparison_Final.png", # 高质量位图备用
    "dpi": 600
}

# 专业出版级绘图样式配置
PLOT_STYLES = {
    "E2E-MAML (Conventional)":      {"color": "#a6cee3", "linestyle": "--", "zorder": 2},
    "Pretrain-MAML (Conventional)": {"color": "#1f78b4", "linestyle": ":", "zorder": 2},
    "E2E-MAML (Functional)":        {"color": "#b2df8a", "linestyle": "-", "zorder": 3},
    "Pretrain-MAML (Functional)":   {"color": "#33a02c", "linestyle": "-", "linewidth": 2.0, "zorder": 4} 
}

# 滑动窗口平均的窗口大小
SMOOTHING_WINDOW = 5

def setup_publication_quality_matplotlib():
    """为Matplotlib设置符合期刊出版质量的全局参数"""
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 8
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    matplotlib.rcParams['lines.linewidth'] = 1.2
    matplotlib.rcParams['axes.linewidth'] = 0.8
    matplotlib.rcParams['xtick.major.width'] = 0.6
    matplotlib.rcParams['ytick.major.width'] = 0.6
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def main():
    """主函数，用于生成最终的对比图"""
    print("开始生成MAML性能对比图 (最终出版级)...")
    
    # --- 2. 设置绘图环境 ---
    setup_publication_quality_matplotlib()
    
    fig, ax = plt.subplots(figsize=(3.3, 2.8)) 

    best_overall_dice = -1
    best_overall_info = {}

    # --- 3. 循环读取、平滑并绘图 ---
    for name, path in FILE_PATHS.items():
        if not os.path.exists(path):
            print(f"警告: 找不到文件 '{path}', 跳过 '{name}'.")
            continue
        
        try:
            df = pd.read_csv(path)
            style = PLOT_STYLES.get(name, {})
            
            df['val_meta_dice_smooth'] = df['val_meta_dice'].rolling(
                window=SMOOTHING_WINDOW, center=True, min_periods=1
            ).mean()

            ax.plot(
                df['epoch'], 
                df['val_meta_dice_smooth'], 
                color=style.get("color"),
                linestyle=style.get("linestyle"),
                linewidth=style.get("linewidth", 1.2),
                label=name,
                zorder=style.get("zorder", 1)
            )
            
            best_idx = df['val_meta_dice'].idxmax()
            current_best_epoch = df.loc[best_idx, 'epoch']
            current_best_dice = df.loc[best_idx, 'val_meta_dice']
            
            if current_best_dice > best_overall_dice:
                best_overall_dice = current_best_dice
                best_overall_info = {
                    "epoch": current_best_epoch,
                    "dice": current_best_dice,
                    "name": name,
                    "color": style.get("color")
                }
        except Exception as e:
            print(f"错误: 处理文件 '{path}' 失败: {e}")
            
    if not any(ax.get_lines()):
        print("错误: 未能成功绘制任何曲线。"); plt.close(fig); return

    # --- 4. 突出显示全局最佳性能点 ---
    if best_overall_info:
        epoch = best_overall_info["epoch"]
        dice = best_overall_info["dice"]
        
        ax.plot(epoch, dice, 'o', markersize=5, color=best_overall_info["color"], 
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        
        # 【修改】使用更优雅的弧线箭头和注释框
        ax.annotate(f'Best: {dice:.4f}',
                    xy=(epoch, dice),
                    xytext=(epoch - 5, dice + 0.08), # 调整文本位置以获得更好效果
                    ha='right', va='center',
                    fontsize=8,
                    arrowprops=dict(
                        arrowstyle='->', # 简洁的箭头样式
                        facecolor='black',
                        connectionstyle="arc3,rad=-0.2", # 这是实现弧线的关键！
                        lw=0.8
                    ),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6)
                   )
        
    # --- 5. 设置图表样式和标签 ---
    # 【修改】将长标题拆分为两行，以避免超出范围
    ax.set_title("MAML Performance\nAcross Task Paradigms", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Meta-Dice Score")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', frameon=False) 
    ax.grid(True, which='major', linestyle=':', linewidth='0.5', color='#cccccc')
    
    # --- 6. 保存图表 ---
    output_path_pdf = OUTPUT_CONFIG["save_path_pdf"]
    output_path_png = OUTPUT_CONFIG["save_path_png"]
    os.makedirs(os.path.dirname(output_path_pdf), exist_ok=True)
    
    # 【修改】在保存前使用 fig.tight_layout() 确保所有元素可见
    # rect参数为总标题留出空间
    fig.tight_layout(rect=[0, 0, 1, 0.92]) 
    
    # 【修改】直接保存为PNG，并确保透明背景和高DPI
    plt.savefig(
        output_path_png, 
        dpi=600, 
        transparent=True # 使背景透明
    )
    plt.savefig(output_path_pdf) # 同时保存一份PDF
    
    plt.close(fig)
    print(f"\n出版级对比图已成功保存到: {output_path_png} (PNG) 和 {output_path_pdf} (PDF)")

if __name__ == "__main__":
    main()