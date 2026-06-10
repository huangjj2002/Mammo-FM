"""
DST-Prototype feature-training launcher.

Modify the variables below and run:
    python run_dst_proto_features.py
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ---- Embedding input ----
EMBEDDING_DIR = r"./output/origin_embeddings_finetuned_fold0"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.csv"

# ---- Output ----
MODEL_SAVE_DIR = r"./best_model/dst_proto_features_run_1"
CSV_OUTPUT_DIR = r"./output/dst_proto_features_run_1"
OUTPUT_DIR = CSV_OUTPUT_DIR
FOLD_CSV = "metadata_folds_dst_proto_run1.csv"
USE_EXISTING_FOLD_CSV = "n"

# ---- Split/data ----
OVERLAP_POLICY = "test"
SPLIT_BY_COHORT = "y"
COHORT_COL = "cohort_num"
TRAIN_COHORTS = "1-8"
TEST_COHORTS = "9-10"
LABEL = "cancer"

# ---- Training ----
N_FOLDS = 0
KFOLD0_VAL_FRAC = 0.2
KFOLD0_VAL_MAX_FRAC = 0.5
EPOCHS = 25
EARLY_STOP = 3
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 8
LR = 5e-5
WEIGHT_DECAY = 1e-4
WEIGHTED_BCE = "y"
WARMUP_EPOCHS = 1
SEED = 42
NUM_WORKERS = 4

# ---- Device ----
DEVICE = "cuda"
GPU_ID = 0
APEX = "y"
PRINT_FREQ = 50
LOG_FREQ = 200

# ---- DST Prototype ----
DST_PROTO_K = 10
DST_PROTO_TOPK = 3
DST_PROTO_NORMALIZE = "y"
DST_PROTO_GAMMA_INIT = 1.0
DST_PROTO_ALPHA_INIT = 0.0
DST_PROTO_DROPOUT = 0.0
DST_PROTO_ATTRACT_WEIGHT = 0.1
DST_PROTO_SEPARATION_WEIGHT = 0.1
DST_PROTO_DIVERSITY_WEIGHT = 0.01
DST_PROTO_LOSS_WEIGHT = 1.0
DST_PROTO_MARGIN = 1.0
DST_PROTO_BALANCE_CLASSES = "y"


RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def append_timestamp_path(path_str, timestamp):
    path_obj = Path(path_str)
    return str(path_obj.with_name(f"{path_obj.name}_{timestamp}"))


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Launch DST-Prototype training from pre-extracted embeddings."
    )
    parser.add_argument("--gpu-id", default=GPU_ID, type=int)
    parser.add_argument("--embedding-dir", default=EMBEDDING_DIR, type=str)
    parser.add_argument("--overlap-policy", default=OVERLAP_POLICY, choices=["error", "test", "train", "training"])
    parser.add_argument("--kfold0-val-frac", default=KFOLD0_VAL_FRAC, type=float)
    parser.add_argument("--kfold0-val-max-frac", default=KFOLD0_VAL_MAX_FRAC, type=float)
    parser.add_argument("--dst-proto-loss-weight", default=DST_PROTO_LOSS_WEIGHT, type=float)
    return parser.parse_args()


def build_args(cli_args):
    model_save_dir = append_timestamp_path(MODEL_SAVE_DIR, RUN_TIMESTAMP)
    csv_output_dir = append_timestamp_path(CSV_OUTPUT_DIR, RUN_TIMESTAMP)
    return argparse.Namespace(
        embedding_dir=cli_args.embedding_dir,
        embeddings_file=EMBEDDINGS_FILE,
        metadata_file=METADATA_FILE,
        model_save_dir=model_save_dir,
        csv_output_dir=csv_output_dir,
        output_dir=csv_output_dir,
        fold_csv=FOLD_CSV,
        use_existing_fold_csv=USE_EXISTING_FOLD_CSV,
        overlap_policy=cli_args.overlap_policy,
        split_by_cohort=SPLIT_BY_COHORT,
        cohort_col=COHORT_COL,
        train_cohorts=TRAIN_COHORTS,
        test_cohorts=TEST_COHORTS,
        label=LABEL,
        n_folds=N_FOLDS,
        kfold0_val_frac=cli_args.kfold0_val_frac,
        kfold0_val_max_frac=cli_args.kfold0_val_max_frac,
        epochs=EPOCHS,
        early_stop=EARLY_STOP,
        batch_size=BATCH_SIZE,
        micro_batch_size=MICRO_BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        weighted_BCE=WEIGHTED_BCE,
        warmup_epochs=WARMUP_EPOCHS,
        seed=SEED,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        gpu_id=cli_args.gpu_id,
        apex=APEX,
        print_freq=PRINT_FREQ,
        log_freq=LOG_FREQ,
        dst_proto_k=DST_PROTO_K,
        dst_proto_topk=DST_PROTO_TOPK,
        dst_proto_normalize=DST_PROTO_NORMALIZE,
        dst_proto_gamma_init=DST_PROTO_GAMMA_INIT,
        dst_proto_alpha_init=DST_PROTO_ALPHA_INIT,
        dst_proto_dropout=DST_PROTO_DROPOUT,
        dst_proto_attract_weight=DST_PROTO_ATTRACT_WEIGHT,
        dst_proto_separation_weight=DST_PROTO_SEPARATION_WEIGHT,
        dst_proto_diversity_weight=DST_PROTO_DIVERSITY_WEIGHT,
        dst_proto_loss_weight=cli_args.dst_proto_loss_weight,
        dst_proto_margin=DST_PROTO_MARGIN,
        dst_proto_balance_classes=DST_PROTO_BALANCE_CLASSES,
        max_samples=None,
    )


if __name__ == "__main__":
    cli_args = parse_cli_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu_id)
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_dst_proto_features import main as _main

    args = build_args(cli_args)
    print("=" * 60)
    print("  DST-Prototype Feature Training")
    print("=" * 60)
    print(f"Selected GPU ID: {args.gpu_id}")
    print(f"Embedding Dir:   {args.embedding_dir}")
    print(f"Run Timestamp:   {RUN_TIMESTAMP}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Model Save Dir:  {args.model_save_dir}")
    print("=" * 60)
    _main(args)
