# 文件: preprocessing_v2.py

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

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_final_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 核心配置 ---
CONFIG = {
    # 为这次决定性的实验创建一个全新的输出目录
    "output_base_folder": "/root/autodl-tmp/best_images_multicenter_nobg_alldices_3s5q", 
    "data_folder": "/root/autodl-tmp/ATLAS_data/ATLAS_2/Training/",
    "t1w_suffix": "_space-MNI152NLin2009aSym_T1w.nii.gz",
    "mask_suffix_pattern": "_space-MNI152NLin2009aSym_label-{hemi}_desc-T1lesion_mask.nii.gz",
    "random_seed": 42,
    
    # 保持与旧脚本一致的处理参数
    "crop_coords": {"x_start": 10, "x_end": 190, "y_start": 40, "y_end": 220 },
    "target_size": (192, 192),

    # --- 切片筛选策略：只保留病变，且不设上限 ---
    "slice_selection_in_preprocess": {
        # 阈值可以设为0.0001或一个极小的正数，以过滤掉几乎没有病变的切片
        # 旧脚本中 mask_rejection_threshold=0 实际上意味着保留所有 lesion_ratio >= 0 的切片
        # 为了更实用，我们用一个极小值来筛掉噪声
        "min_lesion_ratio_to_keep": 0.0001, 
    },

    # --- 中心划分参数 ---
    "center_size_thresholds": {"large": 20, "medium": 10},
    "total_centers": {"train": 25, "val": 8 },

    # --- 元学习任务创建参数 ---
    "support_set_size": 3,
    "query_set_size": 5,
    "max_tasks_per_center": 30,
    # 因为数据只包含病变，不再需要背景分层
    "task_lesion_strata_definitions": {
        "large_lesion":  (0.01, 1.01, 2),
        "medium_lesion": (0.001, 0.01, 3),
        "small_lesion":  (1e-6, 0.001, 3), # 匹配筛选阈值
    },
    "min_total_lesion_slices_in_task": 1, 
}

# --- 所有函数 ---

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def extract_center_id(filename_or_path):
    # (这个函数保持不变)
    basename = os.path.basename(filename_or_path)
    match_r = re.search(r'sub-r(\d+)s\d+', basename)
    if match_r: return f"r{match_r.group(1)}"
    match_direct = re.search(r'sub-([a-zA-Z0-9]+?)_ses', basename)
    if match_direct:
        subject_part = match_direct.group(1)
        if subject_part.startswith('r') and subject_part[1:].isdigit(): return subject_part[:4]
        return subject_part
    path_parts = filename_or_path.split(os.sep)
    for part in reversed(path_parts):
        if part.startswith(('R', 'r')) and len(part) >= 4 and part[1:4].isdigit():
            return part[1:4].lower()
    return "unknown_center"

def analyze_centers(image_files):
    # (这个函数保持不变)
    center_subjects = defaultdict(set)
    for image_file in image_files:
        center_id = extract_center_id(image_file)
        subject_match = re.search(r'(sub-[a-zA-Z0-9]+)', os.path.basename(image_file))
        if subject_match:
            center_subjects[center_id].add(subject_match.group(1))
    center_sizes = {}
    for center, subjects in center_subjects.items():
        n_subjects = len(subjects)
        cat = "small"
        if n_subjects >= CONFIG["center_size_thresholds"]["large"]: cat = "large"
        elif n_subjects >= CONFIG["center_size_thresholds"]["medium"]: cat = "medium"
        center_sizes[center] = {"n_subjects": n_subjects, "size_category": cat}
    return center_sizes

def split_centers_stratified(center_stats_map):
    # (这个函数保持不变)
    all_centers = list(center_stats_map.keys())
    random.shuffle(all_centers)
    num_train = CONFIG["total_centers"]["train"]
    num_val = CONFIG["total_centers"]["val"]
    if len(all_centers) < num_train + num_val:
        if len(all_centers) > 0:
            prop_train = num_train / (num_train + num_val) if (num_train + num_val) > 0 else 0
            num_train = int(len(all_centers) * prop_train)
    train_centers = all_centers[:num_train]
    val_centers = all_centers[num_train : num_train + num_val]
    return {"train": train_centers, "val": val_centers}

def print_split_stats(center_stats, split_centers):
    # (这个函数保持不变)
    logger.info(f"训练中心数: {len(split_centers['train'])}")
    logger.info(f"验证中心数: {len(split_centers['val'])}")

def process_and_save_scan(args):
    """
    处理单个3D扫描：筛选切片，应用切片级归一化，并保存。
    """
    image_file, mask_file, output_dir_base, split_name, config, slice_info_collector = args
    try:
        img_nii = nib.load(image_file)
        img_data_3d = img_nii.get_fdata().astype(np.float32)
        mask_data_3d = nib.load(mask_file).get_fdata().astype(np.uint8)
        center_id = extract_center_id(image_file)

        images_save_dir = os.path.join(output_dir_base, split_name, "images")
        masks_save_dir = os.path.join(output_dir_base, split_name, "masks")
        os.makedirs(images_save_dir, exist_ok=True)
        os.makedirs(masks_save_dir, exist_ok=True)

        min_lr_to_keep = config["slice_selection_in_preprocess"]["min_lesion_ratio_to_keep"]
        
        num_slices_processed = 0
        for i in range(img_data_3d.shape[2]):
            mask_slice = mask_data_3d[:, :, i]
            
            if mask_slice.size == 0 or (np.count_nonzero(mask_slice) / mask_slice.size) < min_lr_to_keep:
                continue

            slice_data = img_data_3d[:, :, i]
            
            crop_cfg = config["crop_coords"]
            slice_cropped = slice_data[crop_cfg["x_start"]:crop_cfg["x_end"], crop_cfg["y_start"]:crop_cfg["y_end"]]
            mask_cropped = mask_slice[crop_cfg["x_start"]:crop_cfg["x_end"], crop_cfg["y_start"]:crop_cfg["y_end"]]
            
            img_resized = cv2.resize(slice_cropped, config["target_size"], interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_cropped, config["target_size"], interpolation=cv2.INTER_NEAREST)
            
            # --- 核心修改：切片级归一化 ---
            slice_mean = np.mean(img_resized)
            slice_std = np.std(img_resized)
            if slice_std < 1e-6: slice_std = 1e-6
            img_normalized = (img_resized - slice_mean) / slice_std
            
            base_fname = f"{os.path.splitext(os.path.basename(image_file))[0]}_slice_{i}"
            img_path = os.path.join(images_save_dir, f"{base_fname}.npy")
            mask_path = os.path.join(masks_save_dir, f"{base_fname}.npy")
            
            np.save(img_path, img_normalized.astype(np.float32))
            np.save(mask_path, mask_resized.astype(np.uint8))
            
            ratio = np.count_nonzero(mask_resized) / mask_resized.size
            slice_info_collector[center_id].append({
                "image_path": img_path, "mask_path": mask_path, 
                "lesion_ratio": float(ratio), "center_id": center_id
            })
            num_slices_processed += 1
            
        return {"status": "success", "processed": num_slices_processed}
    except Exception as e:
        logger.error(f"处理文件 {image_file} 时出错: {e}")
        return {"status": "error"}


def create_meta_tasks(center_slice_data_map, split_name, config):
    logger.info(f"为 {split_name} 集创建 MAML 任务...")
    all_tasks = {}
    task_id_counter = 1
    K_SHOT, K_QUERY = config["support_set_size"], config["query_set_size"]
    total_samples = K_SHOT + K_QUERY
    strata_def = config.get("task_lesion_strata_definitions", {})

    for center_id, slices in center_slice_data_map.items():
        if len(slices) < total_samples: continue
        
        strata = defaultdict(list)
        for s in slices:
            lr = s["lesion_ratio"]
            for name, (min_r, max_r, _) in strata_def.items():
                if min_r <= lr < max_r:
                    strata[name].append(s)
                    break
        
        for _ in range(config["max_tasks_per_center"]):
            task_slices = []
            temp_strata = {k: list(v) for k, v in strata.items()}
            
            for name, (_, _, count) in strata_def.items():
                pool = temp_strata.get(name, [])
                random.shuffle(pool)
                taken = pool[:count]
                task_slices.extend(taken)
                for item in taken: pool.remove(item)
                
            if len(task_slices) < total_samples:
                needed = total_samples - len(task_slices)
                remaining_pool = [s for T in temp_strata.values() for s in T]
                random.shuffle(remaining_pool)
                task_slices.extend(remaining_pool[:needed])
            
            if len(task_slices) < total_samples or sum(1 for s in task_slices if s["lesion_ratio"] > 0) < config["min_total_lesion_slices_in_task"]: continue
                
            random.shuffle(task_slices)
            support_set = task_slices[:K_SHOT]
            query_set = task_slices[K_SHOT:]
            
            if len(support_set) < K_SHOT or len(query_set) < K_QUERY: continue
            
            task_key = f"{split_name}_task_{task_id_counter}"
            all_tasks[task_key] = {
                "center_id": center_id,
                "support_set_paths": [s["image_path"] for s in support_set],
                "query_set_paths": [s["image_path"] for s in query_set],
                "support_set_mask_paths": [s["mask_path"] for s in support_set],
                "query_set_mask_paths": [s["mask_path"] for s in query_set],
            }
            task_id_counter += 1
            
    output_path = os.path.join(config["output_base_folder"], f"{split_name}_meta_tasks.json")
    with open(output_path, 'w') as f: json.dump(all_tasks, f, indent=2)
    logger.info(f"{split_name} 的任务定义已保存到: {output_path} (共 {len(all_tasks)} 个任务)")
    return all_tasks

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

def main():
    """主执行函数"""
    set_random_seeds(CONFIG["random_seed"])
    os.makedirs(CONFIG["output_base_folder"], exist_ok=True)
    logger.info(f"所有输出将保存到: {CONFIG['output_base_folder']}")
    
    # 1. 查找文件对
    image_mask_pairs = get_image_mask_pairs_from_data_folder(CONFIG)
    if not image_mask_pairs: return
    
    # 2. 按中心划分病人
    all_image_files = [p[0] for p in image_mask_pairs]
    center_stats = analyze_centers(all_image_files)
    split_centers = split_centers_stratified(center_stats)
    print_split_stats(center_stats, split_centers)
    
    # 3. Pass 2: 处理并保存所有切片
    train_slice_info = defaultdict(list)
    val_slice_info = defaultdict(list)
    
    args_list = []
    for img_path, mask_path in image_mask_pairs:
        center_id = extract_center_id(img_path)
        if center_id in split_centers["train"]:
            args_list.append((img_path, mask_path, CONFIG["output_base_folder"], "train", CONFIG, train_slice_info))
        elif center_id in split_centers["val"]:
            args_list.append((img_path, mask_path, CONFIG["output_base_folder"], "val", CONFIG, val_slice_info))
            
    logger.info(f"准备处理 {len(args_list)} 个3D扫描...")
    for args in tqdm(args_list, desc="Processing scans"):
        process_and_save_scan(args)

    # 4. 创建元学习任务
    create_meta_tasks(train_slice_info, "train", CONFIG)
    create_meta_tasks(val_slice_info, "val", CONFIG)
    
    logger.info("所有预处理步骤完成！")

if __name__ == "__main__":
    main()