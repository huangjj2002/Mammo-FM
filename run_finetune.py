"""
修改下方变量后直接运行即可：
python run_finetune.py
"""

# =============================================================================
# ========================= 配置区域 ==============================
# =============================================================================

# ---- 路径 ----
DATA_DIR         = r"/home/dhao4/workspace/hjj_workspace/data"                  # 数据根目录
IMG_DIR          = "images_png"                  # 图片目录
CSV_FILE         = "/home/dhao4/workspace/hjj_workspace/data/data.csv"    # CSV文件
CLIP_CHK_PT_PATH = r"./model/Mammo-FM_BatmanlabTrained_CLIP.tar"
OUTPUT_DIR       = r"./output/finetune_cancer"
FOLD_CSV         = None                         
OVERLAP_POLICY   = "test"  # error/test/train for patient split overlap
SPLIT_BY_COHORT  = "y"
COHORT_COL       = "cohort_num"
TRAIN_COHORTS    = "1-8"
TEST_COHORTS     = "9-10"


DATASET          = "Custom"                      
DATA_FRAC        = "1.0"                       
LABEL            = "cancer"                      
ARCH             = "breast_clip_det_b5_period_n_ft"  
FREEZE_BACKBONE  = "n"   # "y" = freeze Mammo-FM backbone; "n" = full fine-tuning

N_FOLDS          = 0       # 交叉验证折数
EPOCHS           = 25      # 每折最大训练轮数
EARLY_STOP       = 3       # 早停参数（0=禁用）
BATCH_SIZE       = 8       # 批大小
LR               = 5e-5    # 学习率
WEIGHT_DECAY     = 1e-4    
WARMUP_EPOCHS    = 1       
WEIGHTED_BCE     = "y"     
SEED             = 42     
NUM_WORKERS      = 4       


IMG_SIZE         = [1520, 912]  # 图片尺寸 [高, 宽]


DEVICE           = "cuda"  
GPU_ID           = 0    #设置使用的GPU   
APEX             = "y"     
PRINT_FREQ       = 50      
LOG_FREQ         = 200    


ALPHA            = 10      
SIGMA            = 15      
P                = 1.0     


MEAN             = 0.3089279
STD              = 0.25053555408335154



import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

sys.path.insert(0, str(Path(__file__).parent))

from finetune_cancer import main as _main
import argparse


def build_args():
  
    args = argparse.Namespace(
        data_dir=DATA_DIR,
        img_dir=IMG_DIR,
        csv_file=CSV_FILE,
        clip_chk_pt_path=CLIP_CHK_PT_PATH,
        output_dir=OUTPUT_DIR,
        fold_csv=FOLD_CSV,
        overlap_policy=OVERLAP_POLICY,
        split_by_cohort=SPLIT_BY_COHORT,
        cohort_col=COHORT_COL,
        train_cohorts=TRAIN_COHORTS,
        test_cohorts=TEST_COHORTS,
        dataset=DATASET,
        data_frac=DATA_FRAC,
        label=LABEL,
        arch=ARCH,
        freeze_backbone=FREEZE_BACKBONE,
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
        gpu_id=GPU_ID,
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
    print("=" * 60)
    print("  Mammo-FM Fine-tuning")
    print("=" * 60)
    print(f"Selected GPU ID:      {GPU_ID}")
    print(f"Folds:                {N_FOLDS}")
    print(f"Freeze Backbone:      {FREEZE_BACKBONE}")
    print(f"Split By Cohort:      {SPLIT_BY_COHORT}")
    print(f"Train/Test Cohorts:   {TRAIN_COHORTS} / {TEST_COHORTS} ({COHORT_COL})")
    print(f"Overlap Policy:       {OVERLAP_POLICY}")
    print("=" * 60)
    _main(args)
