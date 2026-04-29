"""
入口脚本 —— 所有配置变量集中在此，方便修改。
修改下方变量后直接运行即可：
    python run_finetune.py
"""

# =============================================================================
# ========================= 配置区域（按需修改） ==============================
# =============================================================================

# ---- 路径 ----
DATA_DIR         = r"/mnt/g/data"                   # 数据根目录
IMG_DIR          = "images_png"                  # 图片目录（相对于 DATA_DIR，也可用绝对路径）
CSV_FILE         = "train_with_test_data_mini.csv"    # CSV文件（相对于 DATA_DIR，也可用绝对路径）
CLIP_CHK_PT_PATH = r"/mnt/g/Mammo_CLIP_PROJECT/Mammo_FM/Mammo-FM/model/Mammo-FM_BatmanlabTrained_CLIP.tar"
OUTPUT_DIR       = r"/mnt/g/Mammo_CLIP_PROJECT/Mammo_FM/Mammo-FM/output/finetune_cancer"
FOLD_CSV         = None                          # fold CSV保存路径，None则自动生成在CSV同目录

# ---- 数据集 / 任务 ----
DATASET          = "Custom"                      # 数据集名称（仅用于日志标识，不影响逻辑）
DATA_FRAC        = "1.0"                         # 训练数据使用比例
LABEL            = "cancer"                      # CSV中的标签列名
ARCH             = "breast_clip_det_b5_period_n_ft"  # "breast_clip_det_b5_period_n_lp"=线性探针, "breast_clip_det_b5_period_n_ft"=全量微调

# ---- 训练 ----
N_FOLDS          = 5       # 交叉验证折数
EPOCHS           = 10      # 每折最大训练轮数
EARLY_STOP       = 5       # 早停耐心值（0=禁用）
BATCH_SIZE       = 2       # 批大小
LR               = 5e-5    # 学习率
WEIGHT_DECAY     = 1e-4    # 权重衰减
WARMUP_EPOCHS    = 1       # 预热轮数
WEIGHTED_BCE     = "n"     # 是否使用加权BCE ("y"/"n")
SEED             = 42      # 随机种子
NUM_WORKERS      = 2       # 数据加载线程数

# ---- 图片 ----
IMG_SIZE         = [1520, 912]  # 图片尺寸 [高, 宽]

# ---- 系统 ----
DEVICE           = "cuda"  # 设备 ("cuda" 或 "cpu")
APEX             = "y"     # 混合精度 ("y"/"n")
PRINT_FREQ       = 50      # 训练日志打印频率（每N步）
LOG_FREQ         = 200     # TensorBoard日志频率（每N步）

# ---- 数据增强 ----
ALPHA            = 10      # ElasticTransform alpha
SIGMA            = 15      # ElasticTransform sigma
P                = 1.0     # 增强概率

# ---- 归一化 ----
MEAN             = 0.3089279
STD              = 0.25053555408335154

# =============================================================================
# =========================== 运行（无需修改） ===============================
# =============================================================================

import sys
from pathlib import Path

# 确保从脚本所在目录运行也能找到模块
sys.path.insert(0, str(Path(__file__).parent))

from finetune_cancer import main as _main
import argparse


def build_args():
    """将上方变量转为 argparse.Namespace 对象。"""
    args = argparse.Namespace(
        data_dir=DATA_DIR,
        img_dir=IMG_DIR,
        csv_file=CSV_FILE,
        clip_chk_pt_path=CLIP_CHK_PT_PATH,
        output_dir=OUTPUT_DIR,
        fold_csv=FOLD_CSV,
        dataset=DATASET,
        data_frac=DATA_FRAC,
        label=LABEL,
        arch=ARCH,
        n_folds=N_FOLDS,
        epochs=EPOCHS,
        early_stop=EARLY_STOP,
        batch_size=BATCH_SIZE,
        micro_batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        weighted_BCE=WEIGHTED_BCE,
        seed=SEED,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
        device=DEVICE,
        apex=APEX,
        print_freq=PRINT_FREQ,
        log_freq=LOG_FREQ,
        alpha=ALPHA,
        sigma=SIGMA,
        p=P,
        mean=MEAN,
        std=STD,
    )
    return args


if __name__ == "__main__":
    args = build_args()
    _main(args)
