import torch
import os

# --- 路径配置 (Path Configuration) ---
# !!! 重要: 请根据你的实际情况更新此路径 !!!
# ROOT_DATA_DIR 应指向包含 train/, val/, test/ 子目录的 'best_images' 文件夹
# --- 路径配置 (Path Configuration) ---
# ROOT_DATA_DIR 应指向包含 train/, val/ 子目录的 'best_images_multicenter' 文件夹
ROOT_DATA_DIR = '/root/autodl-tmp/maml_data_functional_tasks' # 根据你的截图，这是正确的
PRETRAINED_MODEL_PATH = "/root/autodl-tmp/maml_data_functional_tasks_baseline_train/checkpoints/best_model.pth" 

# 定义 train 和 val 目录的名称 (相对于 ROOT_DATA_DIR)
TRAIN_SUBDIR_NAME = "train"
VAL_SUBDIR_NAME = "val"
# TEST_SUBDIR_NAME = "test" # 如果有测试集

# 完整的训练数据根目录 (包含所有 center_rXXX)
TRAIN_TASKS_JSON_PATH = os.path.join(ROOT_DATA_DIR, TRAIN_SUBDIR_NAME, "train_functional_meta_tasks.json")
VAL_TASKS_JSON_PATH = os.path.join(ROOT_DATA_DIR, VAL_SUBDIR_NAME, "val_functional_meta_tasks.json")
# TEST_DATA_ROOT_FOR_CENTERS = os.path.join(ROOT_DATA_DIR, TEST_下 !!!
BASE_OUTPUT_DIR = '/root/autodl-tmp/maml_data_functional_tasks_finetune_maml' # 例输出路径，请修改
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, 'checkpoints_maml')
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'results_maml')
PROCESSED_NPY_ROOT_DIR = ROOT_DATA_DIR
TRAIN_TASKS_JSON_PATH = os.path.join(ROOT_DATA_DIR, "train_functional_meta_tasks.json")
VAL_TASKS_JSON_PATH = os.path.join(ROOT_DATA_DIR, "val_functional_meta_tasks.json")
if not os.path.exists(TRAIN_TASKS_JSON_PATH):
    print(f"警告: 训练任务JSON文件 '{TRAIN_TASKS_JSON_PATH}' 未找到!")
if not os.path.exists(VAL_TASKS_JSON_PATH):
    print(f"警告: 验证任务JSON文件 '{VAL_TASKS_JSON_PATH}' 未找到!")

# --- 设备配置 (Device Configuration) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_IDS = [0] # 用于 DataParallel 的 GPU ID 列表, 例如 [0, 1]。如果只有一块或用CPU，设为 [0] 或 []。
                   # MAML 与 DataParallel 结合可能复杂，建议先单GPU运行。

# --- MAML 超参数 (MAML Hyperparameters) ---
# N_WAY: 对于分割任务，通常 N-way 是 1 (即目标类别只有一种：病变区域)
K_SHOT =  3         # 支持集 (support set) 中每个任务的样本数//原来是8
K_QUERY = 5        # 查询集 (query set) 中每个任务的样本数//原来是15
NUM_TASKS_PER_EPOCH = 200 # 每个元学习 epoch 中采样的任务总数
META_BATCH_SIZE = 4     # 外层循环更新元模型时，一批处理的任务数量//原来是4
                        # 如果显存不足，可以减小此值

INNER_LR = 0.001         # 内层循环的学习率 (用于任务适应)
OUTER_LR = 0.001       # 外层循环的学习率 (用于更新元模型)
NUM_INNER_STEPS = 5     # 内层循环的梯度更新步数

NUM_META_EPOCHS = 100   # 总的元学习训练轮数

# --- 模型超参数 (Model Hyperparameters) ---
MODEL_IN_CHANNELS = 1   # 输入图像的通道数 (例如灰度图为1)
MODEL_CLASSES = 1       # 分割的类别数 (二分类分割，病变vs背景，通常为1，输出一个概率图)
BIFPN_CHANNELS = 64     # BiFPN模块中特征图的通道数

# --- 损失函数超参数 (Loss Function Hyperparameters) ---
BCE_WEIGHT = 0.1        # BCE Loss 在总损失中的权重
EDGE_WEIGHT = 3.0       # 边缘 Dice Loss 在 Dice Loss 中的权重 (原代码为3.0，可调整)

# --- 训练设置 (Training Settings) ---
SEED = 42               # 随机种子，用于可复现性
NUM_WORKERS = 4        # DataLoader 使用的工作进程数 (根据你的CPU核心数调整)
PATIENCE_LR_SCHEDULER = 7 # ReduceLROnPlateau 学习率调度器的耐心值
WEIGHT_DECAY = 1e-4     # 优化器的权重衰减 (L2正则化)

# --- 自动创建输出目录 (Ensure output directories exist) ---SUBDIR_NAME) # 如果有

# !!! 重要: 所有输出（检查点、结果图表）将保存在此基础目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Config Loaded: DEVICE={DEVICE}, ROOT_DATA_DIR='{ROOT_DATA_DIR}'")
print(f"TRAIN_TASKS_JSON_PATH='{TRAIN_TASKS_JSON_PATH}'")
print(f"VAL_TASKS_JSON_PATH='{VAL_TASKS_JSON_PATH}'")

if not os.path.exists(TRAIN_TASKS_JSON_PATH):
    print(f"警告: 训练任务JSON文件 '{TRAIN_TASKS_JSON_PATH}' 未找到!")
if not os.path.exists(VAL_TASKS_JSON_PATH):
    print(f"警告: 验证任务JSON文件 '{VAL_TASKS_JSON_PATH}' 未找到!")