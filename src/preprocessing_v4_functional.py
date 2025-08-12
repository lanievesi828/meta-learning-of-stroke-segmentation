# 文件: src/preprocessing_v4_functional.py
# 版本: 利用Excel元数据，按解剖位置构建功能性任务

import os
import glob
import re
import json
import numpy as np
import nibabel as nib
import cv2
import random
from tqdm import tqdm
import logging
from collections import defaultdict
import pandas as pd

# --- 1. 日志和随机种子配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("preprocessing_functional.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)
def set_random_seeds(seed=42): random.seed(seed); np.random.seed(seed)

# --- 2. 核心配置 ---
CONFIG = {
    "data_folder": "/root/autodl-tmp/ATLAS_data/ATLAS_2/Training/",
    "metadata_file": "/root/autodl-tmp/ATLAS_data/ATLAS_2/20220425_ATLAS_2.0_MetaData.xlsx",
    # 【修改】为这次功能性任务的数据集指定一个全新的目录
    "output_base_folder": "/root/autodl-tmp/maml_data_functional_tasks", 
    
    # 【修改】根据你的Excel输出，更新元数据列名
    "metadata_cols": {
        "subject_id": "Subject ID",
        # 我们需要从这几列推断出主要位置
        "cortical_lh": "# Lesions LH Cortical and White Matter",
        "subcortical_lh": "# Lesions LH Subcortical",
        "cortical_rh": "# Lesions RH Cortical and White Matter",
        "subcortical_rh": "# Lesions RH Subcortical",
        "other": "# Lesions Other Location"
    },

    # --- 其他配置保持不变 ---
    "t1w_suffix": "_space-MNI152NLin2009aSym_T1w.nii.gz",
    "mask_suffix_pattern": "_space-MNI152NLin2009aSym_label-{hemi}_desc-T1lesion_mask.nii.gz",
    "crop_coords": {"x_start": 10, "x_end": 190, "y_start": 40, "y_end": 220 },
    "target_size": (192, 192),
    "random_seed": 42,
    "slice_selection_in_preprocess": {
        "max_lesion_slices_per_volume": 50,
        "max_background_slices_per_volume": 30,
    },
    "support_set_size": 3,
    "query_set_size": 5,
    "train_split_ratio": 0.8 # 按病人划分训练/验证集的比例
}

# --- 3. 辅助函数 ---
# 你之前的 get_image_mask_pairs... 和 process_scan... 函数可以保持原样，我们会在main函数里调用它们
def get_image_mask_pairs_from_data_folder(config):
    # (此函数可以保持原样)
    logger.info("正在查找图像和掩码文件...")
    all_files = glob.glob(os.path.join(config["data_folder"], "**", "*.nii.gz"), recursive=True)
    t1w_files = [f for f in all_files if config["t1w_suffix"] in f]
    pairs = []
    for img_f in tqdm(t1w_files, desc="匹配图像和掩码"):
        base = img_f.replace(config["t1w_suffix"], "")
        mask_l = base + config["mask_suffix_pattern"].replace("{hemi}", "L")
        mask_r = base + config["mask_suffix_pattern"].replace("{hemi}", "R")
        if os.path.exists(mask_l): pairs.append((img_f, mask_l))
        elif os.path.exists(mask_r): pairs.append((img_f, mask_r))
        else: # 尝试通用匹配
            generic_masks = glob.glob(base + "*lesion_mask.nii.gz")
            if generic_masks: pairs.append((img_f, generic_masks[0]))
    logger.info(f"共找到 {len(pairs)} 个图像-掩码对。")
    return pairs

def process_and_collect_slices(image_mask_pairs, metadata_df, output_base_folder, config):
    """
    【最终修正版】处理所有3D扫描，执行实例级归一化和切片筛选，并返回一个带有元数据的切片信息列表。
    已修复 NameError。
    """
    all_slices_info = []
    images_save_dir = os.path.join(output_base_folder, "all_slices", "images")
    masks_save_dir = os.path.join(output_base_folder, "all_slices", "masks")
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(masks_save_dir, exist_ok=True)

    id_col = config["metadata_cols"]["subject_id"]
    cat_col = "lesion_category"
    id_to_category = metadata_df.set_index(id_col)[cat_col].to_dict()
    
    debug_counter = 0

    for image_file, mask_file in tqdm(image_mask_pairs, desc="处理3D扫描并筛选切片"):
        try:
            # --- 提取病人ID并获取其功能类别 ---
            subject_id_match = re.search(r'(sub-r\d+s\d+)', os.path.basename(image_file))
            if not subject_id_match: continue
            subject_id = subject_id_match.group(1)
            
            if subject_id not in id_to_category: continue
            lesion_category = id_to_category[subject_id]

            # --- 加载数据 & 实例级归一化 ---
            img_data_3d = nib.load(image_file).get_fdata().astype(np.float32)
            mask_data_3d = nib.load(mask_file).get_fdata().astype(np.uint8)
            pixels = img_data_3d[img_data_3d > 0]
            norm_mean, norm_std = (np.mean(pixels), np.std(pixels)) if pixels.size > 1 else (0.0, 1.0)
            if norm_std < 1e-6: norm_std = 1e-6
            
            # --- 切片筛选逻辑 ---
            lesion_slice_indices, background_slice_indices = [], []
            for i in range(mask_data_3d.shape[2]):
                if np.any(mask_data_3d[:, :, i] > 0):
                    lesion_slice_indices.append(i)
                else:
                    background_slice_indices.append(i)
            
            # 调试日志
            if debug_counter < 5:
                logger.info(f"\n--- 调试病人: {subject_id} ---")
                logger.info(f"  原始切片数: 总共={mask_data_3d.shape[2]}, 病灶={len(lesion_slice_indices)}, 背景={len(background_slice_indices)}")
            
            random.shuffle(lesion_slice_indices)
            random.shuffle(background_slice_indices)

            cfg_slice = config["slice_selection_in_preprocess"]
            max_lesion = cfg_slice["max_lesion_slices_per_volume"]
            max_bg = cfg_slice["max_background_slices_per_volume"]
            
            selected_lesion = lesion_slice_indices[:max_lesion]
            selected_background = background_slice_indices[:max_bg]
            
            if debug_counter < 5:
                logger.info(f"  筛选上限: 病灶={max_lesion}, 背景={max_bg}")
                logger.info(f"  筛选后切片数: 病灶={len(selected_lesion)}, 背景={len(selected_background)}")
                if len(lesion_slice_indices) > max_lesion or len(background_slice_indices) > max_bg:
                    logger.info("  >>>>> 筛选已生效，原始切片数超过上限! <<<<<")
                else:
                    logger.info("  >>>>> 筛选未起削减作用，原始切片数未达到上限。 <<<<<")
            debug_counter += 1

            selected_indices = sorted(selected_lesion + selected_background)
            if not selected_indices: continue

            # --- 【核心修正】循环处理每一个被筛选出来的切片 ---
            for slice_idx in selected_indices:
                # 图像处理和保存
                slice_data = img_data_3d[:, :, slice_idx]
                mask_slice = mask_data_3d[:, :, slice_idx]
                crop_cfg = config["crop_coords"]
                slice_cropped = slice_data[crop_cfg["x_start"]:crop_cfg["x_end"], crop_cfg["y_start"]:crop_cfg["y_end"]]
                mask_cropped = mask_slice[crop_cfg["x_start"]:crop_cfg["x_end"], crop_cfg["y_start"]:crop_cfg["y_end"]]
                img_resized = cv2.resize(slice_cropped, config["target_size"], interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_cropped, config["target_size"], interpolation=cv2.INTER_NEAREST)
                img_norm = (img_resized - norm_mean) / norm_std
                
                # 定义文件名和路径 (现在 img_path 和 mask_path 在这里被定义)
                base_fname = f"{subject_id}_slice_{slice_idx}"
                img_path = os.path.join(images_save_dir, f"{base_fname}.npy")
                mask_path = os.path.join(masks_save_dir, f"{base_fname}.npy")
                np.save(img_path, img_norm)
                np.save(mask_path, mask_resized)
                
                # 【核心修正】在循环内部收集信息
                # 此时 img_path 和 mask_path 已经被正确定义
                all_slices_info.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "subject_id": subject_id,
                    "lesion_category": lesion_category
                })
        except Exception as e:
            logger.error(f"处理文件 {image_file} 时出错: {e}")
    
    return all_slices_info
def create_functional_meta_tasks(slice_df, split_name, config):
    """根据病灶解剖位置创建功能性元任务"""
    logger.info(f"为 {split_name} 集创建功能性MAML任务...")
    
    # 1. 按功能（解剖位置）对所有切片进行分池
    functional_pools = defaultdict(list)
    for index, row in slice_df.iterrows():
        functional_pools[row['lesion_category']].append(row.to_dict())

    logger.info(f"{split_name} 集功能池大小: " + ", ".join([f"'{k}': {len(v)}" for k, v in functional_pools.items()]))

    # 2. 为每个功能池独立创建任务
    all_tasks = {}
    task_counter = 0
    task_size = config["support_set_size"] + config["query_set_size"]

    for category, pool in functional_pools.items():
        if len(pool) < task_size:
            logger.warning(f"类别 '{category}' 样本不足 ({len(pool)}个)，无法创建任务。")
            continue
        
        random.shuffle(pool)
        
        num_tasks_for_category = len(pool) // task_size
        logger.info(f"正在为类别 '{category}' 创建 {num_tasks_for_category} 个任务...")
        
        for i in range(num_tasks_for_category):
            start_idx = i * task_size
            end_idx = start_idx + task_size
            task_slices = pool[start_idx:end_idx]
            
            support_set = task_slices[:config["support_set_size"]]
            query_set = task_slices[config["support_set_size"]:]
            
            all_tasks[f"task_{task_counter}"] = {
                "support_set_paths": [s["image_path"] for s in support_set],
                "support_set_mask_paths": [s["mask_path"] for s in support_set],
                "query_set_paths": [s["image_path"] for s in query_set],
                "query_set_mask_paths": [s["mask_path"] for s in query_set],
                "task_category": category
            }
            task_counter += 1

    logger.info(f"为 {split_name} 集成功创建了 {len(all_tasks)} 个功能性任务。")
    output_path = os.path.join(config["output_base_folder"], f"{split_name}_functional_meta_tasks.json")
    with open(output_path, 'w') as f:
        json.dump(all_tasks, f, indent=2)
    logger.info(f"{split_name} 功能性任务定义已保存到: {output_path}")

# --- 4. 主执行函数 ---
def main():
    set_random_seeds(CONFIG["random_seed"])
    os.makedirs(CONFIG["output_base_folder"], exist_ok=True)
    
    # 1. 加载和处理元数据 (和之前一样)
    logger.info(f"加载元数据 from {CONFIG['metadata_file']}...")
    metadata_df = pd.read_excel(CONFIG['metadata_file'])
    cols = CONFIG["metadata_cols"]
    def get_lesion_category(row):
        num_cortical = row[cols["cortical_lh"]] + row[cols["cortical_rh"]]
        num_subcortical = row[cols["subcortical_lh"]] + row[cols["subcortical_rh"]]
        num_other = row[cols["other"]]
        if num_cortical > 0: return "Cortical"
        elif num_subcortical > 0: return "Subcortical"
        elif num_other > 0: return "Other"
        else: return "No_Lesion_Documented"
    loc_cols = [cols["cortical_lh"], cols["subcortical_lh"], cols["cortical_rh"], cols["subcortical_rh"], cols["other"]]
    metadata_df[loc_cols] = metadata_df[loc_cols].fillna(0)
    metadata_df["lesion_category"] = metadata_df.apply(get_lesion_category, axis=1)
    logger.info("元数据处理完成。各类别病人数量:\n" + metadata_df["lesion_category"].value_counts().to_string())

    # 2. 查找所有图像-掩码对 (和之前一样)
    image_mask_pairs = get_image_mask_pairs_from_data_folder(CONFIG)
    if not image_mask_pairs: return

    # 3. 【调用新函数】处理所有扫描并执行切片筛选
    all_slices_info = process_and_collect_slices(image_mask_pairs, metadata_df, CONFIG["output_base_folder"], CONFIG)
    all_slices_df = pd.DataFrame(all_slices_info)
    logger.info(f"所有扫描处理完成，共生成 {len(all_slices_df)} 个经过筛选的2D切片。")

    # 4. 按病人ID划分训练/验证集 (和之前一样)
    all_subjects = all_slices_df['subject_id'].unique()
    random.shuffle(all_subjects)
    split_idx = int(len(all_subjects) * CONFIG["train_split_ratio"])
    train_subjects = set(all_subjects[:split_idx])
    train_slice_df = all_slices_df[all_slices_df['subject_id'].isin(train_subjects)]
    val_slice_df = all_slices_df[~all_slices_df['subject_id'].isin(train_subjects)]
    logger.info(f"数据集划分为: {len(train_slice_df)} 个训练切片, {len(val_slice_df)} 个验证切片。")

    # 5. 为训练集和验证集分别创建功能性任务 (和之前一样)
    create_functional_meta_tasks(train_slice_df, "train", CONFIG)
    create_functional_meta_tasks(val_slice_df, "val", CONFIG)

    logger.info("所有预处理步骤完成！")

if __name__ == "__main__":
    main()