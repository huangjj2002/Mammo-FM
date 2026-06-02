"""
Prototype + EDL fine-tuning launcher.

Modify the variables below and run:
    python run_proto_edl.py

CLI override example:
    python finetune_proto_edl.py --data-dir /path/to/data --csv-file data.csv --edl-proto-k 4
"""

# =============================================================================
# ========================= Configuration ==========================
# =============================================================================

# ---- Paths ----
DATA_DIR = r"/home/dhao4/workspace/hjj_workspace/data"
IMG_DIR = "images_png"
CSV_FILE = "/home/dhao4/workspace/hjj_workspace/data/data.csv"
CLIP_CHK_PT_PATH = r"./model/Mammo-FM_BatmanlabTrained_CLIP.tar"
MODEL_SAVE_DIR = r"./best_model/proto_run_1"
CSV_OUTPUT_DIR = r"./output/proto_run_1"
OUTPUT_DIR = CSV_OUTPUT_DIR
FOLD_CSV = "data_folds_proto_edl_run1.csv"
USE_EXISTING_FOLD_CSV = "n"

# ---- Data ----
OVERLAP_POLICY = "test"
SPLIT_BY_COHORT = "y"
COHORT_COL = "cohort_num"
TRAIN_COHORTS = "1-8"
TEST_COHORTS = "9-10"

DATASET = "Custom"
DATA_FRAC = "1.0"
LABEL = "cancer"
ARCH = "breast_clip_det_b5_period_n_ft"
FREEZE_BACKBONE = "n"

# ---- Training ----
N_FOLDS = 0
KFOLD0_VAL_FRAC = 0.2
KFOLD0_VAL_MAX_FRAC = 0.5
EPOCHS = 25
EARLY_STOP = 3
BATCH_SIZE = 8
LR = 5e-5
WEIGHT_DECAY = 1e-4
WEIGHTED_BCE = "y"
WARMUP_EPOCHS = 1
SEED = 42
NUM_WORKERS = 4

# ---- Image ----
IMG_SIZE = [1520, 912]

# ---- Device ----
DEVICE = "cuda"
GPU_ID = 0
APEX = "y"
PRINT_FREQ = 50
LOG_FREQ = 200

# ---- Augmentation ----
ALPHA = 10
SIGMA = 15
P = 1.0

# ---- Normalization ----
MEAN = 0.3089279
STD = 0.25053555408335154

# ---- EDL ----
EVIDENCE_TYPE = "softplus"
EDL_LOSS_TYPE = "log"
EDL_KL_WEIGHT = 0.1
ANNEALING_COEF = EDL_KL_WEIGHT
ANNEALING_START = 0.0

# ---- Prototype ----
EDL_PROTO_K = 10
EDL_PROTO_TOPK = 3
EDL_PROTO_TEMPERATURE = 1.0
EDL_PROTO_NORMALIZE = "y"
EDL_PROTO_CLASS_WEIGHT = 1.0
EDL_PROTO_ATTRACT_WEIGHT = 0.1
EDL_PROTO_SEPARATION_WEIGHT = 0.1
EDL_PROTO_DIVERSITY_WEIGHT = 0.01
EDL_PROTO_LOSS_WEIGHT = 1.0
EDL_PROTO_MARGIN = 1.0
EDL_PROTO_BALANCE_CLASSES = "y"


import os
import sys
from pathlib import Path
import argparse

ANNEALING_STEP = EPOCHS


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Launch Prototype + EDL fine-tuning with optional runtime overrides."
    )
    parser.add_argument("--gpu-id", default=GPU_ID, type=int, help=f"GPU device ID (default: {GPU_ID})")
    parser.add_argument(
        "--overlap-policy",
        default=OVERLAP_POLICY,
        choices=["error", "test", "train", "training"],
        help="How to handle patients present in both training and test splits.",
    )
    parser.add_argument(
        "--kfold0-val-frac",
        default=KFOLD0_VAL_FRAC,
        type=float,
        help=(
            "Only used when N_FOLDS=0. Fraction of the training pool held out "
            f"as validation (default: {KFOLD0_VAL_FRAC}). Values > 1 are treated as percent."
        ),
    )
    parser.add_argument(
        "--kfold0-val-max-frac",
        default=KFOLD0_VAL_MAX_FRAC,
        type=float,
        help=(
            "Only used when N_FOLDS=0. Maximum validation fraction when expanding "
            f"a single-class validation split (default: {KFOLD0_VAL_MAX_FRAC})."
        ),
    )
    parser.add_argument(
        "--edl-proto-loss-weight",
        "--edl_proto_loss_weight",
        dest="edl_proto_loss_weight",
        default=EDL_PROTO_LOSS_WEIGHT,
        type=float,
        help=f"Global multiplier applied to prototype regularization (default: {EDL_PROTO_LOSS_WEIGHT}).",
    )
    return parser.parse_args()


def build_args(
    gpu_id=None,
    overlap_policy=None,
    kfold0_val_frac=None,
    kfold0_val_max_frac=None,
    edl_proto_loss_weight=None,
):
    resolved_gpu_id = GPU_ID if gpu_id is None else int(gpu_id)
    resolved_overlap_policy = OVERLAP_POLICY if overlap_policy is None else overlap_policy
    resolved_kfold0_val_frac = KFOLD0_VAL_FRAC if kfold0_val_frac is None else float(kfold0_val_frac)
    resolved_kfold0_val_max_frac = (
        KFOLD0_VAL_MAX_FRAC if kfold0_val_max_frac is None else float(kfold0_val_max_frac)
    )
    resolved_proto_loss_weight = (
        EDL_PROTO_LOSS_WEIGHT if edl_proto_loss_weight is None else float(edl_proto_loss_weight)
    )
    return argparse.Namespace(
        data_dir=DATA_DIR,
        img_dir=IMG_DIR,
        csv_file=CSV_FILE,
        clip_chk_pt_path=CLIP_CHK_PT_PATH,
        model_save_dir=MODEL_SAVE_DIR,
        csv_output_dir=CSV_OUTPUT_DIR,
        output_dir=OUTPUT_DIR,
        fold_csv=FOLD_CSV,
        use_existing_fold_csv=USE_EXISTING_FOLD_CSV,
        overlap_policy=resolved_overlap_policy,
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
        kfold0_val_frac=resolved_kfold0_val_frac,
        kfold0_val_max_frac=resolved_kfold0_val_max_frac,
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
        img_size=IMG_SIZE,
        device=DEVICE,
        gpu_id=resolved_gpu_id,
        apex=APEX,
        print_freq=PRINT_FREQ,
        log_freq=LOG_FREQ,
        alpha=ALPHA,
        sigma=SIGMA,
        p=P,
        mean=MEAN,
        std=STD,
        evidence_type=EVIDENCE_TYPE,
        edl_loss_type=EDL_LOSS_TYPE,
        edl_kl_weight=EDL_KL_WEIGHT,
        annealing_coef=ANNEALING_COEF,
        annealing_step=ANNEALING_STEP,
        annealing_start_frac=ANNEALING_START,
        edl_proto_k=EDL_PROTO_K,
        edl_proto_topk=EDL_PROTO_TOPK,
        edl_proto_temperature=EDL_PROTO_TEMPERATURE,
        edl_proto_normalize=EDL_PROTO_NORMALIZE,
        edl_proto_class_weight=EDL_PROTO_CLASS_WEIGHT,
        edl_proto_attract_weight=EDL_PROTO_ATTRACT_WEIGHT,
        edl_proto_separation_weight=EDL_PROTO_SEPARATION_WEIGHT,
        edl_proto_diversity_weight=EDL_PROTO_DIVERSITY_WEIGHT,
        edl_proto_loss_weight=resolved_proto_loss_weight,
        edl_proto_margin=EDL_PROTO_MARGIN,
        edl_proto_balance_classes=EDL_PROTO_BALANCE_CLASSES,
    )


if __name__ == "__main__":
    cli_args = parse_cli_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu_id)

    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_proto_edl import main as _main

    args = build_args(
        gpu_id=cli_args.gpu_id,
        overlap_policy=cli_args.overlap_policy,
        kfold0_val_frac=cli_args.kfold0_val_frac,
        kfold0_val_max_frac=cli_args.kfold0_val_max_frac,
        edl_proto_loss_weight=cli_args.edl_proto_loss_weight,
    )
    print("=" * 60)
    print("  Prototype + Evidential Deep Learning (EDL) Fine-tuning")
    print("=" * 60)
    print(f"Selected GPU ID:      {cli_args.gpu_id}")
    print(f"EDL Loss Type:        {EDL_LOSS_TYPE}")
    print(f"EDL KL Weight:        {EDL_KL_WEIGHT}")
    print(f"EDL Annealing Step:   {ANNEALING_STEP}")
    print(f"EDL Annealing Start:  {ANNEALING_START}")
    print(f"Freeze Backbone:      {FREEZE_BACKBONE}")
    print(f"Weighted BCE/Data:    {WEIGHTED_BCE}")
    print(f"Prototype K:          {EDL_PROTO_K}")
    print(f"Prototype TopK:       {EDL_PROTO_TOPK}")
    print(f"Prototype Temp:       {EDL_PROTO_TEMPERATURE}")
    print(f"Prototype Normalize:  {EDL_PROTO_NORMALIZE}")
    print(f"Prototype Class Wt:   {EDL_PROTO_CLASS_WEIGHT}")
    print(f"Prototype Attract Wt: {EDL_PROTO_ATTRACT_WEIGHT}")
    print(f"Prototype Separate Wt:{EDL_PROTO_SEPARATION_WEIGHT}")
    print(f"Prototype Diversity Wt:{EDL_PROTO_DIVERSITY_WEIGHT}")
    print(f"Prototype Loss Wt:    {cli_args.edl_proto_loss_weight}")
    print(f"Prototype Margin:     {EDL_PROTO_MARGIN}")
    print(f"Prototype Balance Cls:{EDL_PROTO_BALANCE_CLASSES}")
    print(f"KFold0 Val Frac:      {cli_args.kfold0_val_frac}")
    print(f"KFold0 Val Max Frac:  {cli_args.kfold0_val_max_frac}")
    print(f"Overlap Policy:       {cli_args.overlap_policy}")
    print(f"Split By Cohort:      {SPLIT_BY_COHORT}")
    print(f"Train/Test Cohorts:   {TRAIN_COHORTS} / {TEST_COHORTS} ({COHORT_COL})")
    print(f"Use Existing Fold CSV: {USE_EXISTING_FOLD_CSV}")
    print(f"Model Save Dir:       {MODEL_SAVE_DIR}")
    print(f"CSV Output Dir:       {CSV_OUTPUT_DIR}")
    print("=" * 60)
    _main(args)
