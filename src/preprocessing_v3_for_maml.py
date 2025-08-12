# 文件: preprocessing_v3_for_maml_final.py
# 版本: 实例级归一化 + 按中心构建任务 (健壮版)

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

# --- 1. 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_maml_heterogeneous_final.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. 核心配置 ---
CONFIG = {
    "data_folder": "/root/autodl-tmp/ATLAS_data/ATLAS_2/Training/",
    "output_base_folder": "/root/autodl-tmp/maml_data_heterogeneous_3s5q",
    "t1w_suffix": "_space-MNI152NLin2009aSym_T1w.nii.gz",
    "mask_suffix_pattern": "_space-MNI152NLin2009aSym_label-{hemi}_desc-T1lesion_mask.nii.gz",
    
    "crop_coords": {"x_start": 10, "x_end": 190, "y_start": 40, "y_end": 220 },
    "target_size": (192, 192),
    "random_seed": 42,
    
    "center_size_thresholds": {"large": 20, "medium": 10},
    "total_centers": {"train": 25, "val": 8 },

    "slice_selection_in_preprocess": {
        "min_lesion_ratio_to_keep_any_lesion": 0.0001, 
        "max_lesion_slices_per_volume": 50,
        "max_background_slices_per_volume": 30,
    },

    "support_set_size": 3,
    "query_set_size": 5,
    "max_tasks_per_center": 30,

    # 【重要】确保这个定义的总和与 support_set_size + query_set_size 匹配
    "task_lesion_strata_definitions": {
        "large_lesion":  (0.01, 1.01, 2),    # (min_ratio, max_ratio, num_samples_to_draw)
        "medium_lesion": (0.001, 0.01, 2),
        "small_lesion":  (0.0001, 0.001, 2),
        "background":    (0.0, 0.0001, 2),
    }, # 总和: 2+2+2+2 = 8, 恰好等于 3-shot + 5-query
    
    "min_total_lesion_slices_in_task": 2, 
}

# --- 3. 所有函数 ---

# set_random_seeds, extract_center_id, analyze_centers, 
# split_centers_stratified, print_split_stats, get_image_mask_pairs_from_data_folder
# 这些辅助函数保持不变，这里省略以保持简洁，请从你之前的代码复制。
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


def process_scan_instance_norm(args):
    # (此函数保持不变，它已经足够健壮)
    image_file, mask_file, output_dir_base, split_name, config, collector = args
    try:
        img_nii = nib.load(image_file); img_data_3d = img_nii.get_fdata().astype(np.float32)
        mask_data_3d = nib.load(mask_file).get_fdata().astype(np.uint8); center_id = extract_center_id(image_file)
        pixels = img_data_3d[img_data_3d > 0]
        norm_mean, norm_std = (np.mean(pixels), np.std(pixels)) if pixels.size > 1 else (0.0, 1.0)
        if norm_std < 1e-6: norm_std = 1e-6
        images_save_dir = os.path.join(output_dir_base, split_name, "images"); masks_save_dir = os.path.join(output_dir_base, split_name, "masks")
        os.makedirs(images_save_dir, exist_ok=True); os.makedirs(masks_save_dir, exist_ok=True)
        cfg = config["slice_selection_in_preprocess"]; lesion_slices, bg_slices = [], []
        for i in range(img_data_3d.shape[2]):
            mask_slice = mask_data_3d[:, :, i]; ratio = np.count_nonzero(mask_slice) / mask_slice.size if mask_slice.size > 0 else 0
            if ratio >= cfg["min_lesion_ratio_to_keep_any_lesion"]: lesion_slices.append(i)
            else: bg_slices.append(i)
        random.shuffle(lesion_slices); random.shuffle(bg_slices)
        selected_indices = sorted(lesion_slices[:cfg["max_lesion_slices_per_volume"]] + bg_slices[:cfg["max_background_slices_per_volume"]])
        for slice_idx in selected_indices:
            slice_data = img_data_3d[:, :, slice_idx]; mask_slice = mask_data_3d[:, :, slice_idx]; crop_cfg = config["crop_coords"]
            slice_cropped = slice_data[crop_cfg["x_start"]:crop_cfg["x_end"], crop_cfg["y_start"]:crop_cfg["y_end"]]
            mask_cropped = mask_slice[crop_cfg["x_start"]:crop_cfg["x_end"], crop_cfg["y_start"]:crop_cfg["y_end"]]
            img_resized = cv2.resize(slice_cropped, config["target_size"], interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_cropped, config["target_size"], interpolation=cv2.INTER_NEAREST)
            img_norm = (img_resized - norm_mean) / norm_std
            base_fname = f"{os.path.splitext(os.path.basename(image_file))[0]}_slice_{slice_idx}"
            img_path = os.path.join(images_save_dir, f"{base_fname}.npy"); mask_path = os.path.join(masks_save_dir, f"{base_fname}.npy")
            np.save(img_path, img_norm); np.save(mask_path, mask_resized)
            ratio = np.count_nonzero(mask_resized) / mask_resized.size if mask_resized.size > 0 else 0
            collector[center_id].append({"image_path": img_path, "mask_path": mask_path, "lesion_ratio": float(ratio)})
        return {"status": "success", "processed": len(selected_indices)}
    except Exception as e:
        logger.error(f"处理文件 {image_file} 时出错: {e}"); return {"status": "error"}

def create_meta_tasks_per_center(center_slice_data_map, split_name, config):
    """
    【最终优化版】按中心创建元任务。
    能健壮地处理样本不足的情况，并提供清晰的日志。
    """
    logger.info(f"为 {split_name} 集创建 按中心划分的 MAML 任务...")
    all_tasks = {}
    task_counter = 0
    total_required_per_task = config["support_set_size"] + config["query_set_size"]

    # 遍历每一个中心
    for center_id, slices in tqdm(center_slice_data_map.items(), desc=f"为 {split_name} 创建任务"):
        
        # 1. 检查该中心是否有足够的总切片数来创建至少一个任务
        if len(slices) < total_required_per_task:
            logger.warning(f"跳过中心 {center_id}，因为它只有 {len(slices)} 个切片，少于任务所需的 {total_required_per_task} 个。")
            continue
        
        # 2. 对该中心的所有切片按病灶比例进行分层
        strata = defaultdict(list)
        for s in slices:
            # 使用一个辅助函数来确定分层，更清晰
            stratum_name = get_stratum_name(s["lesion_ratio"], config)
            strata[stratum_name].append(s)

        # 3. 在这个中心内，尝试创建多个任务
        center_task_count = 0
        for _ in range(config["max_tasks_per_center"]):
            
            # 【核心逻辑】检查每个分层是否都有足够的样本可供抽取
            is_possible_to_create_task = True
            for stratum_name, (_, _, num_to_draw) in config["task_lesion_strata_definitions"].items():
                if len(strata[stratum_name]) < num_to_draw:
                    is_possible_to_create_task = False
                    break # 只要有一个分层样本不够，就无法创建这个任务
            
            # 如果样本不足，就停止为该中心创建更多任务
            if not is_possible_to_create_task:
                if center_task_count == 0:
                    logger.warning(f"中心 {center_id} 虽然总样本足够，但其分层样本不足，无法创建任何一个符合规格的任务。")
                break
            
            # 如果样本充足，就从每个分层中【不放回地】抽取样本
            task_slices = []
            temp_strata_for_task = {k: v[:] for k, v in strata.items()} # 创建一个临时副本
            
            for stratum_name, (_, _, num_to_draw) in config["task_lesion_strata_definitions"].items():
                # 从副本中抽取
                drawn_samples = random.sample(temp_strata_for_task[stratum_name], num_to_draw)
                task_slices.extend(drawn_samples)
                # 【重要】从原始分层中移除已用样本，确保不重复使用
                for sample in drawn_samples:
                    strata[stratum_name].remove(sample)

            # 检查任务是否满足最小病灶切片数要求
            lesion_slice_count = sum(1 for s in task_slices if s["lesion_ratio"] > config["slice_selection_in_preprocess"]["min_lesion_ratio_to_keep_any_lesion"])
            if lesion_slice_count < config["min_total_lesion_slices_in_task"]:
                continue # 不满足要求，跳过这个任务组合，进行下一次尝试

            # 创建 support set 和 query set
            random.shuffle(task_slices)
            support_set = task_slices[:config["support_set_size"]]
            query_set = task_slices[config["support_set_size"]:]

            # 最终确认数量无误后，保存任务
            if len(support_set) == config["support_set_size"] and len(query_set) == config["query_set_size"]:
                all_tasks[f"task_{task_counter}"] = {
                    "support_set_paths": [s["image_path"] for s in support_set],
                    "support_set_mask_paths": [s["mask_path"] for s in support_set],
                    "query_set_paths": [s["image_path"] for s in query_set],
                    "query_set_mask_paths": [s["mask_path"] for s in query_set],
                    "center_id": center_id
                }
                task_counter += 1
                center_task_count += 1
        
        if center_task_count > 0:
            logger.info(f"为中心 {center_id} 创建了 {center_task_count} 个任务。")

    logger.info(f"为 {split_name} 集成功创建了 {len(all_tasks)} 个任务。")
    output_path = os.path.join(config["output_base_folder"], f"{split_name}_meta_tasks.json")
    with open(output_path, 'w') as f:
        json.dump(all_tasks, f, indent=2)
    logger.info(f"{split_name} 任务定义已保存到: {output_path}")

def get_stratum_name(ratio, config):
    """一个辅助函数，根据病灶比例返回其所属的分层名称"""
    for stratum_name, (min_r, max_r, _) in config["task_lesion_strata_definitions"].items():
        if min_r <= ratio < max_r:
            return stratum_name
    # 处理边界情况，如果一个病灶比例特别大，归入最后一个分层
    if ratio >= list(config["task_lesion_strata_definitions"].values())[-1][1]:
        return list(config["task_lesion_strata_definitions"].keys())[-1]
    return "unknown_stratum"


def main():
    """主执行函数"""
    set_random_seeds(CONFIG["random_seed"])
    os.makedirs(CONFIG["output_base_folder"], exist_ok=True)
    logger.info(f"所有输出将保存到: {CONFIG['output_base_folder']}")
    
    image_mask_pairs = get_image_mask_pairs_from_data_folder(CONFIG)
    if not image_mask_pairs: return
    
    all_image_files = [p[0] for p in image_mask_pairs]
    center_stats = analyze_centers(all_image_files)
    split_centers = split_centers_stratified(center_stats)
    print_split_stats(center_stats, split_centers)
    
    train_slice_info = defaultdict(list)
    val_slice_info = defaultdict(list)
    
    all_args = []
    for img_path, mask_path in image_mask_pairs:
        center_id = extract_center_id(img_path)
        collector, split = (train_slice_info, "train") if center_id in train_slice_info or center_id in split_centers["train"] else \
                           (val_slice_info, "val") if center_id in val_slice_info or center_id in split_centers["val"] else (None, None)
        if collector is not None:
            all_args.append((img_path, mask_path, CONFIG["output_base_folder"], split, CONFIG, collector))

    logger.info(f"准备处理 {len(all_args)} 个3D扫描...")
    for args in tqdm(all_args, desc="Processing Scans"):
        process_scan_instance_norm(args)
        
    create_meta_tasks_per_center(train_slice_info, "train", CONFIG)
    create_meta_tasks_per_center(val_slice_info, "val", CONFIG)
    
    logger.info("所有预处理步骤完成！")

if __name__ == "__main__":
    main()