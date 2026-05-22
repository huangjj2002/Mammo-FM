"""
EDL (Evidential Deep Learning) 微调启动脚本

修改下方变量后直接运行即可：
    python run_edl.py

功能说明:
    - 基于 Mammo-FM 骨干网络，使用 EDL 模块进行不确定性估计
    - 五折交叉验证，通过 FREEZE_BACKBONE 显式控制是否冻结骨干网络
    - 损失函数: EDL 类交叉熵损失 + KL 散度正则化（带退火）
    - 训练结束自动调用测试模块
    - 输出 CSV 包含: fold, evidence_0, evidence_1, alpha_0, alpha_1, uncertainty, pred_score, pred_label 等
"""

# =============================================================================
# ========================= 配置区域 ==============================
# =============================================================================

# ---- 路径 ----
DATA_DIR         = r"/home/dhao4/workspace/hjj_workspace/data"                   # 数据根目录
IMG_DIR          = "images_png"                  # 图片目录
CSV_FILE         = "/home/dhao4/workspace/hjj_workspace/data/data.csv"    # CSV文件
#DATA_DIR         = r"G:\data"                   # 数据根目录
#IMG_DIR          = r"images_png"                  # 图片目录
#CSV_FILE         = r"train_with_test_data_mini.csv"    # CSV文件
CLIP_CHK_PT_PATH = r"./model/Mammo-FM_BatmanlabTrained_CLIP.tar"
MODEL_SAVE_DIR   = r"./best_model"              # 最佳模型保存目录
CSV_OUTPUT_DIR   = r"./output"                  # CSV 输出目录
OUTPUT_DIR       = CSV_OUTPUT_DIR               # 兼容旧参数
FOLD_CSV         = None                          # None = 自动生成

# ---- 数据 ----
OVERLAP_POLICY   = "test"  # error/test/train for patient split overlap
SPLIT_BY_COHORT  = "y"     # y = use cohort_num to create train/test split
COHORT_COL       = "cohort_num"
TRAIN_COHORTS    = "1-8"
TEST_COHORTS     = "9-10"

DATASET          = "Custom"
DATA_FRAC        = "1.0"
LABEL            = "cancer"
ARCH             = "breast_clip_det_b5_period_n_ft"  # 全量微调
FREEZE_BACKBONE  = "y"   # "y" = freeze Mammo-FM backbone and train only EDL head; "n" = full fine-tuning

# ---- 训练 ----
N_FOLDS          = 0       # 交叉验证折数
EPOCHS           = 25      # 每折最大训练轮数
EARLY_STOP       = 3       # 早停参数（0=禁用）
BATCH_SIZE       = 8      # 批大小
LR               = 5e-5    # 学习率
WEIGHT_DECAY     = 1e-4
WEIGHTED_BCE     = "y"      # EDL 中仅对 data loss/CE 部分做类别加权
WARMUP_EPOCHS    = 1
SEED             = 42
NUM_WORKERS      = 4

# ---- 图片 ----
IMG_SIZE         = [1520, 912]  # 图片尺寸 [高, 宽]

# ---- 设备 ----
DEVICE           = "cuda"
GPU_ID           = 4
APEX             = "y"
PRINT_FREQ       = 50
LOG_FREQ         = 200

# ---- 数据增强 ----
ALPHA            = 10       # ElasticTransform alpha
SIGMA            = 15       # ElasticTransform sigma
P                = 1.0      # 增强概率

# ---- 归一化 ----
MEAN             = 0.3089279
STD              = 0.25053555408335154

# ---- EDL 特有参数 ----
EVIDENCE_TYPE    = "softplus"   # evidence 激活函数: "relu", "exp", "softplus"
EDL_LOSS_TYPE    = "log"        # EDL 损失类型: "log"(类交叉熵), "digamma", "mse"
EDL_KL_WEIGHT    = 0.1          # KL 散度 lambda 权重
ANNEALING_COEF   = EDL_KL_WEIGHT  # 兼容旧参数名
ANNEALING_START  = 0.0          # KL 退火开始比例（0.0=从头开始，0.5=训练一半后开始）


# =============================================================================
# ========================= 运行逻辑 ==============================
# =============================================================================

import os
import sys
from pathlib import Path
import argparse

ANNEALING_STEP = EPOCHS


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Launch EDL fine-tuning with optional runtime overrides."
    )
    parser.add_argument(
        "--gpu-id",
        default=GPU_ID,
        type=int,
        help=f"GPU device ID to use for training (default: {GPU_ID})",
    )
    parser.add_argument(
        "--overlap-policy",
        default=OVERLAP_POLICY,
        choices=["error", "test", "train", "training"],
        help="How to handle patients present in both training and test splits.",
    )
    return parser.parse_args()


def build_args(gpu_id=None, overlap_policy=None):
    """构建参数 Namespace"""
    resolved_gpu_id = GPU_ID if gpu_id is None else int(gpu_id)
    resolved_overlap_policy = OVERLAP_POLICY if overlap_policy is None else overlap_policy
    args = argparse.Namespace(
        # 路径
        data_dir=DATA_DIR,
        img_dir=IMG_DIR,
        csv_file=CSV_FILE,
        clip_chk_pt_path=CLIP_CHK_PT_PATH,
        model_save_dir=MODEL_SAVE_DIR,
        csv_output_dir=CSV_OUTPUT_DIR,
        output_dir=OUTPUT_DIR,
        fold_csv=FOLD_CSV,
        overlap_policy=resolved_overlap_policy,
        split_by_cohort=SPLIT_BY_COHORT,
        cohort_col=COHORT_COL,
        train_cohorts=TRAIN_COHORTS,
        test_cohorts=TEST_COHORTS,
        # 数据
        dataset=DATASET,
        data_frac=DATA_FRAC,
        label=LABEL,
        arch=ARCH,
        freeze_backbone=FREEZE_BACKBONE,
        # 训练
        n_folds=N_FOLDS,
        epochs=EPOCHS,
        early_stop=EARLY_STOP,
        batch_size=BATCH_SIZE,
        micro_batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        weighted_BCE=WEIGHTED_BCE,
        warmup_epochs=WARMUP_EPOCHS,
        seed=SEED,
        num_workers=NUM_WORKERS,
        # 图片
        img_size=IMG_SIZE,
        # 设备
        device=DEVICE,
        gpu_id=resolved_gpu_id,
        apex=APEX,
        print_freq=PRINT_FREQ,
        log_freq=LOG_FREQ,
        # 数据增强
        alpha=ALPHA,
        sigma=SIGMA,
        p=P,
        # 归一化
        mean=MEAN,
        std=STD,
        # EDL 特有参数
        evidence_type=EVIDENCE_TYPE,
        edl_loss_type=EDL_LOSS_TYPE,
        edl_kl_weight=EDL_KL_WEIGHT,
        annealing_coef=ANNEALING_COEF,
        annealing_step=ANNEALING_STEP,
        annealing_start_frac=ANNEALING_START,
    )
    return args


if __name__ == "__main__":
    cli_args = parse_cli_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu_id)

    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_edl import main as _main

    args = build_args(gpu_id=cli_args.gpu_id, overlap_policy=cli_args.overlap_policy)
    print("=" * 60)
    print("  Evidential Deep Learning (EDL) Fine-tuning")
    print("=" * 60)
    print(f"Selected GPU ID:      {cli_args.gpu_id}")
    print(f"EDL Evidence Type:    {EVIDENCE_TYPE}")
    print(f"EDL Loss Type:        {EDL_LOSS_TYPE}")
    print(f"EDL KL Weight:        {EDL_KL_WEIGHT}")
    print(f"EDL Annealing Step:   {ANNEALING_STEP}")
    print(f"EDL Annealing Start:  {ANNEALING_START}")
    print(f"Freeze Backbone:      {FREEZE_BACKBONE}")
    print(f"Weighted BCE/Data:    {WEIGHTED_BCE}")
    print(f"Overlap Policy:       {cli_args.overlap_policy}")
    print(f"Split By Cohort:      {SPLIT_BY_COHORT}")
    print(f"Train/Test Cohorts:   {TRAIN_COHORTS} / {TEST_COHORTS} ({COHORT_COL})")
    print(f"Model Save Dir:       {MODEL_SAVE_DIR}")
    print(f"CSV Output Dir:       {CSV_OUTPUT_DIR}")
    print("=" * 60)
    _main(args)
