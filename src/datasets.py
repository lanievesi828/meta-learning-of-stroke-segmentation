# 文件: src/datasets.py

from collections import defaultdict
import logging
from torch.utils.data import Sampler
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random

# 导入 albumentations 以支持数据增强
import albumentations as A

class MedicalImageSegmentationDataset(Dataset):
    """
    加载并（可选地）增强单个医学图像及其对应的掩码。
    这个类现在是数据增强的主要执行者。
    """
    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): 一个来自于 albumentations 的变换管道。
        """
        self.transform = transform

    def __len__(self):
        # 这个类的长度没有实际意义，因为它被当作工具按需调用
        return 1

    def load_single_sample(self, image_path, mask_path):
        """
        加载并预处理单个图像和掩码对。
        Args:
            image_path (str): 图像文件的完整路径。
            mask_path (str): 掩码文件的完整路径。
        Returns:
            dict or None: 包含 'image' 和 'mask' 张量的字典，或加载失败时返回None。
        """
        try:
            # 加载Numpy数组，此时它们应该是 (H, W) 形状
            img_np = np.load(image_path) 
            msk_np = np.load(mask_path)
        except Exception as e:
            print(f"错误: MedicalImageSegmentationDataset - 无法加载文件 {image_path} 或 {mask_path}. 错误: {e}")
            return None
        
        # *** 核心修改：在这里应用数据增强 ***
        # albumentations 在 (H, W, C) 或 (H, W) 的Numpy数组上工作
        if self.transform:
            try:
                augmented = self.transform(image=img_np, mask=msk_np)
                img_np = augmented['image']
                msk_np = augmented['mask']
            except Exception as e:
                print(f"警告: MedicalImageSegmentationDataset - 应用 transform 到样本 {os.path.basename(image_path)} 时失败. 错误: {e}")
                # 如果增强失败，我们将继续使用原始图像
                pass

        # 在所有变换之后，统一处理数据格式：增加通道维度 -> 转为Tensor -> 确保类型正确
        img_np = np.expand_dims(img_np, axis=0)  # 变为 (1, H, W)
        msk_np = np.expand_dims(msk_np, axis=0)  # 变为 (1, H, W)

        image_tensor = torch.from_numpy(img_np.astype(np.float32))
        mask_tensor = torch.from_numpy(msk_np.astype(np.float32))
        
        # 确保掩码是二值的
        mask_tensor = (mask_tensor > 0.5).float()
            
        sample = {'image': image_tensor, 'mask': mask_tensor}
        return sample

    def __getitem__(self, idx):
        # 这个方法不应该被直接调用
        raise NotImplementedError("请使用 load_single_sample(img_path, msk_path) 方法。")


class FunctionalTaskBatchSampler(Sampler):
    """
    一个自定义的采样器，确保DataLoader返回的每个batch中的所有任务
    都来自同一个功能类别 (如 'Cortical', 'Subcortical')。
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        super().__init__(dataset)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 【修改】从按中心分组改为按功能类别分组
        self.category_to_task_indices = self.dataset.get_category_to_task_indices()
        self.categories = list(self.category_to_task_indices.keys())
        
        if not self.categories:
            raise ValueError("数据集中未找到任何有效的任务类别，无法使用FunctionalTaskBatchSampler。")
        
        self.logger.info(f"FunctionalTaskBatchSampler: 找到 {len(self.categories)} 个功能类别，总共 {len(self.dataset)} 个任务。")
        for category, indices in self.category_to_task_indices.items():
            self.logger.info(f"  - 类别 '{category}': {len(indices)} 个任务")

        # 计算总批次数
        self.num_batches = 0
        for category in self.categories:
            num_tasks_in_category = len(self.category_to_task_indices[category])
            if self.drop_last:
                self.num_batches += num_tasks_in_category // self.batch_size
            else:
                self.num_batches += (num_tasks_in_category + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = []
        categories = self.categories[:]
        if self.shuffle:
            random.shuffle(categories)

        # 【修改】遍历每个功能类别，而不是中心
        for category in categories:
            indices = self.category_to_task_indices[category][:]
            if self.shuffle:
                random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if len(batch) > 0:
                     batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        
        return iter(batches)

    def __len__(self):
        return self.num_batches

# ==============================================================================
# 【核心修改】MetaTaskDataset
# ==============================================================================
class MetaTaskDataset(Dataset):
    """
    从JSON文件加载MAML任务定义，并增加了获取功能类别信息的功能。
    """
    def __init__(self, tasks_json_path, transform=None):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tasks_json_path = tasks_json_path
        self.sample_loader = MedicalImageSegmentationDataset(transform=transform)

        self.logger.info(f"MetaTaskDataset: 正在从 '{self.tasks_json_path}' 加载任务定义...")
        try:
            with open(self.tasks_json_path, 'r') as f:
                tasks_definitions_dict = json.load(f)
            # 保持任务顺序的稳定性
            self.task_items = sorted(list(tasks_definitions_dict.items()))
            if not self.task_items:
                raise ValueError(f"JSON文件 '{self.tasks_json_path}' 中没有定义任何任务。")
            
            transform_info = "已应用" if transform is not None else "无"
            self.logger.info(f"MetaTaskDataset: 从JSON加载了 {len(self.task_items)} 个任务定义。数据增强: {transform_info}")

        except Exception as e:
            raise RuntimeError(f"无法加载或解析任务JSON文件 '{self.tasks_json_path}': {e}")
            
        # 【修改】为新的采样器准备数据
        self._prepare_category_mapping()

    def _prepare_category_mapping(self):
        """
        【修改】创建一个从功能类别到任务索引列表的映射。
        """
        self.category_to_task_indices = defaultdict(list)
        for i, (task_name, task_def) in enumerate(self.task_items):
            # 【修改】现在我们查找 'task_category' 字段
            category = task_def.get("task_category")
            if category:
                self.category_to_task_indices[category].append(i)
            else:
                self.logger.warning(f"任务 '{task_name}' 缺少 'task_category' 字段。")
    
    def get_category_to_task_indices(self):
        """
        【修改】一个公共接口，供新的采样器调用。
        """
        return self.category_to_task_indices

    def __len__(self):
        return len(self.task_items)

    def __getitem__(self, index):
        # 这个函数的逻辑保持不变
        task_name, task_def = self.task_items[index]
        support_images, support_masks = self._load_set(task_def, "support", task_name)
        query_images, query_masks = self._load_set(task_def, "query", task_name)
        if not support_images or not query_images:
             raise RuntimeError(f"未能为任务 '{task_name}' 构建完整的支持集或查询集。")
        return {
            'support_images': torch.stack(support_images),
            'support_masks': torch.stack(support_masks),
            'query_images': torch.stack(query_images),
            'query_masks': torch.stack(query_masks)
        }

    def _load_set(self, task_def, set_type, task_name):
        # 这个辅助函数保持不变
        image_filepaths = task_def.get(f"{set_type}_set_paths", [])
        mask_filepaths = task_def.get(f"{set_type}_set_mask_paths", [])
        images, masks = [], []
        if len(image_filepaths) != len(mask_filepaths):
            raise ValueError(f"任务 '{task_name}' 的 {set_type} 集路径数量不匹配。")
        for img_fp, msk_fp in zip(image_filepaths, mask_filepaths):
            sample = self.sample_loader.load_single_sample(img_fp, msk_fp)
            if sample is None:
                raise RuntimeError(f"无法加载任务 '{task_name}' 中的 {set_type} 样本: img='{img_fp}'")
            images.append(sample['image'])
            masks.append(sample['mask'])
        return images, masks
