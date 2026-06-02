"""
Evidential Deep Learning (EDL) 微调脚本

基于 Mammo-FM 骨干网络，使用 EDL 模块进行五折交叉验证训练和测试。
不修改原项目代码，完全独立脚本。

功能:
    - 五折交叉验证训练
    - EDL 类交叉熵损失 + KL 散度正则化
    - 通过 --freeze-backbone 显式控制是否冻结骨干网络
    - 训练结束自动调用测试模块
    - 输出 CSV 包含: fold, evidence_0, evidence_1, uncertainty, pred_score, pred_label 等

用法:
    python run_edl.py
    或
    python finetune_edl.py --data-dir /path/to/data --clip_chk_pt_path /path/to/checkpoint.tar
"""

import sys

if "--help" in sys.argv or "-h" in sys.argv:
    import argparse
    _p = argparse.ArgumentParser(
        description="5-Fold CV fine-tuning with Evidential Deep Learning (EDL) for breast cancer detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _p.add_argument("--data-dir", default=r"G:\data", type=str,
                    help=r"Root data directory (default: G:\data)")
    _p.add_argument("--img-dir", default="images_png", type=str,
                    help="Image directory, relative to --data-dir (default: images_png)")
    _p.add_argument("--csv-file", default="train_with_test_data.csv", type=str,
                    help="CSV data file, relative to --data-dir (default: train_with_test_data.csv)")
    _p.add_argument("--clip_chk_pt_path",
                    default="./model/Mammo-FM_BatmanlabTrained_CLIP.tar",
                    type=str, help="Path to pre-trained Mammo-FM checkpoint (.tar)")
    _p.add_argument("--model-save-dir", default="./best_model", type=str,
                    help="Directory for best checkpoints and tensorboard logs (default: ./best_model)")
    _p.add_argument("--csv-output-dir", default="./output", type=str,
                    help="Directory for fold and prediction CSV files (default: ./output)")
    _p.add_argument("--output-dir", default=None, type=str,
                    help="Legacy output directory. Used only when new output dirs are not set.")
    _p.add_argument("--fold-csv", default=None, type=str,
                    help="Path to save the CSV with fold column")
    _p.add_argument("--use-existing-fold-csv", default="n", type=str,
                    help="Use --fold-csv as-is and skip fold generation/overwrite (default: n)")
    _p.add_argument("--overlap-policy", default="test",
                    choices=["error", "test", "train", "training"],
                    help="How to handle patients present in both train and test splits (default: test)")
    _p.add_argument("--split-by-cohort", default="y", type=str,
                    help="Use cohort column to create train/test split (default: y)")
    _p.add_argument("--cohort-col", default="cohort_num", type=str,
                    help="Cohort column name (default: cohort_num)")
    _p.add_argument("--train-cohorts", default="1-8", type=str,
                    help="Comma/range cohort spec for training pool (default: 1-8)")
    _p.add_argument("--test-cohorts", default="9-10", type=str,
                    help="Comma/range cohort spec for test set (default: 9-10)")

    _p.add_argument("--dataset", default="Custom", type=str, help="Dataset name (default: Custom)")
    _p.add_argument("--data_frac", default="1.0", type=str, help="Fraction of training data (default: 1.0)")
    _p.add_argument("--label", default="cancer", type=str, help="Label column name (default: cancer)")
    _p.add_argument("--arch", default="breast_clip_det_b5_period_n_ft",
                    choices=["breast_clip_det_b5_period_n_lp", "breast_clip_det_b5_period_n_ft"],
                    help="Architecture (default: breast_clip_det_b5_period_n_ft)")
    _p.add_argument("--freeze-backbone", default="n", choices=["y", "n"],
                    help="Freeze Mammo-FM image encoder and train only the EDL head (default: n)")

    _p.add_argument("--n_folds", default=5, type=int, help="Number of CV folds. 0 disables CV (default: 5)")
    _p.add_argument("--kfold0-val-frac", default=0.2, type=float,
                    help="Only when --n_folds 0: fraction of training pool held out as validation. "
                         "Values > 1 are treated as percent (default: 0.2)")
    _p.add_argument("--kfold0-val-max-frac", default=0.5, type=float,
                    help="Only when --n_folds 0: max validation fraction when expanding a single-class val split. "
                         "Values > 1 are treated as percent (default: 0.5)")
    _p.add_argument("--epochs", default=10, type=int, help="Max epochs per fold (default: 10)")
    _p.add_argument("--early-stop", default=5, type=int, help="Early stopping patience (default: 5)")
    _p.add_argument("--batch-size", default=4, type=int, help="Batch size (default: 4)")
    _p.add_argument("--micro-batch-size", default=1, type=int,
                    help="GPU micro-batch size (default: 1)")
    _p.add_argument("--lr", default=5e-5, type=float, help="Learning rate (default: 5e-5)")
    _p.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay (default: 1e-4)")
    _p.add_argument("--weighted_BCE", "--weighted-bce", dest="weighted_BCE", default="y", type=str,
                    help="Legacy class-balance switch. In EDL it weights only the data loss/CE term (default: y)")
    _p.add_argument("--warmup-epochs", default=1, type=float, help="Warmup epochs (default: 1)")
    _p.add_argument("--img-size", nargs="+", default=[1520, 912], type=int, help="Image size [H W] (default: 1520 912)")
    _p.add_argument("--seed", default=42, type=int, help="Random seed (default: 42)")
    _p.add_argument("--num-workers", default=2, type=int, help="DataLoader num_workers (default: 2)")

    _p.add_argument("--gpu-id", default=0, type=int, help="GPU device ID (default: 0)")
    _p.add_argument("--device", default="cuda", type=str, help="Device: cuda or cpu (default: cuda)")
    _p.add_argument("--apex", default="y", type=str, help="AMP mixed precision: y/n (default: y)")
    _p.add_argument("--print-freq", default=50, type=int, help="Print frequency (default: 50)")
    _p.add_argument("--log-freq", default=200, type=int, help="TensorBoard log frequency (default: 200)")

    _p.add_argument("--alpha", default=10, type=float, help="ElasticTransform alpha (default: 10)")
    _p.add_argument("--sigma", default=15, type=float, help="ElasticTransform sigma (default: 15)")
    _p.add_argument("--p", default=1.0, type=float, help="Augmentation probability (default: 1.0)")

    _p.add_argument("--mean", default=0.3089279, type=float, help="Image normalization mean (default: 0.3089279)")
    _p.add_argument("--std", default=0.25053555408335154, type=float, help="Image normalization std (default: 0.25053555408335154)")

    # EDL specific parameters
    _p.add_argument("--evidence-type", default="softplus", choices=["relu", "exp", "softplus"],
                    help="Evidence activation function (default: softplus)")
    _p.add_argument("--edl-loss-type", default="log", choices=["log", "digamma", "mse"],
                    help="EDL loss type: log=cross-entropy-like, digamma, mse (default: log)")
    _p.add_argument("--annealing-coef", default=0.1, type=float,
                    help="Legacy alias for KL divergence lambda weight (default: 0.1)")
    _p.add_argument("--edl-kl-weight", default=None, type=float,
                    help="KL divergence lambda weight. Overrides --annealing-coef when set")
    _p.add_argument("--annealing-step", default=None, type=float,
                    help="KL annealing step in epochs. Uses lambda=min(1,(epoch+1)/step)*kl_weight when set")
    _p.add_argument("--annealing-start-frac", default=0.0, type=float,
                    help="Legacy fallback: fraction of total epochs before starting KL annealing (default: 0.0)")

    _p.parse_args(["--help"])


import argparse
import gc
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations import Compose, HorizontalFlip, VerticalFlip, Affine, ElasticTransform, Resize
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DTYPES = {
    "patient_id": str,
    "image_id": str,
    "split": str,
}


def read_mammo_csv(csv_path):
    """Read mammo CSV while preserving identifier columns as strings."""
    return pd.read_csv(csv_path, dtype=CSV_DTYPES)


def resolve_project_path(path_value, default_value=None, base_dir=PROJECT_ROOT):
    """Resolve a path relative to the project root unless it is absolute."""
    value = default_value if path_value in (None, "") else path_value
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute() and not os.path.isabs(str(path)):
        path = base_dir / path
    return path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src" / "codebase"))

from metrics import all_classification_metrics, compute_opt_thres
from utils import seed_all, AverageMeter, timeSince
from breastclip.scheduler import LinearWarmupCosineAnnealingLR

# 导入 EDL 模块
from edl_model import MammoEDLModel
from edl_loss import EDLLogLossWithAnnealing, get_evidence


# ==================== Dataset ====================

class CustomMammoDataset(Dataset):
    """Dataset for breast cancer classification from custom CSV + PNG images."""
    def __init__(self, df, img_root, label_col="cancer", transform=None,
                 mean=0.3089279, std=0.25053555408335154):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.label_col = label_col
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / str(row["patient_id"]) / str(row["image_id"])
        img = Image.open(str(img_path)).convert("RGB")
        img = np.array(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        img = img.astype("float32")
        img -= img.min()
        max_val = img.max()
        if max_val > 0:
            img /= max_val
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        label = float(row[self.label_col])
        return {
            "x": img.unsqueeze(0),
            "y": torch.tensor(label, dtype=torch.float32),
            "img_path": str(img_path),
        }


def collate_fn(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.tensor([item["y"].item() for item in batch], dtype=torch.float32),
        "img_path": [item["img_path"] for item in batch],
    }


# ==================== Augmentation ====================

def get_train_transform(img_size=(1520, 912), alpha=10, sigma=15, p=1.0):
    return Compose([
        Resize(height=img_size[0], width=img_size[1]),
        Compose([
            HorizontalFlip(),
            VerticalFlip(),
            Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
            ElasticTransform(alpha=alpha, sigma=sigma),
        ], p=p),
    ])


def get_val_transform(img_size=(1520, 912)):
    return Compose([
        Resize(height=img_size[0], width=img_size[1]),
    ])


# ==================== Early Stopping ====================

class EarlyStopping:
    """Stop training when validation AUROC has not improved for `patience` epochs."""
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0

    def __call__(self, val_auroc):
        if val_auroc > self.best_score:
            self.best_score = val_auroc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def reset(self):
        self.best_score = -float("inf")
        self.counter = 0


# ==================== Fold Creation ====================

def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_cohort_spec(value):
    if isinstance(value, (list, tuple, set)):
        tokens = []
        for item in value:
            tokens.extend(str(item).split(","))
    else:
        tokens = str(value).split(",")

    cohorts = set()
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"Invalid cohort range {token!r}: end < start.")
            cohorts.update(range(start, end + 1))
        else:
            cohorts.add(int(token))
    if not cohorts:
        raise ValueError(f"Cohort spec {value!r} did not contain any cohort numbers.")
    return cohorts


def resolve_cohort_column(df, cohort_col):
    if cohort_col in df.columns:
        return cohort_col
    for candidate in ("cohort_num", "cohert_num"):
        if candidate in df.columns:
            print(f"[Split] WARNING: cohort column {cohort_col!r} not found. Using {candidate!r}.")
            return candidate
    raise ValueError(f"CSV is missing cohort column {cohort_col!r}.")


def normalize_split_values(split_series):
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    normalized = split_series.astype(str).str.strip().str.lower().map(aliases)
    if normalized.isna().any():
        bad = sorted(set(split_series[normalized.isna()].astype(str).str.strip()))
        raise ValueError(f"Unsupported split value(s): {bad}. Expected train/training, val/valid, or test.")
    return normalized


def normalize_fraction_arg(value, name):
    frac = float(value)
    if frac > 1.0:
        frac /= 100.0
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, or between 0 and 100 as a percent; got {value!r}.")
    return frac


def assign_kfold0_validation_split(df, train_mask, label_col, val_frac, val_max_frac, seed):
    """Split the train pool into train/val for n_folds=0 while keeping patients grouped."""
    val_frac = normalize_fraction_arg(val_frac, "kfold0_val_frac")
    val_max_frac = normalize_fraction_arg(val_max_frac, "kfold0_val_max_frac")
    if val_max_frac < val_frac:
        raise ValueError(
            f"kfold0_val_max_frac ({val_max_frac:.4f}) must be >= "
            f"kfold0_val_frac ({val_frac:.4f})."
        )

    train_df = df[train_mask].copy()
    patient_info = train_df.groupby("patient_id").agg(
        label=(label_col, "max"),
        n_images=("image_id", "count"),
    ).reset_index()
    patient_info["label"] = patient_info["label"].astype(int)
    n_patients = len(patient_info)
    if n_patients < 2:
        raise ValueError("Need at least 2 training patients to create a validation split when n_folds=0.")

    rng = np.random.RandomState(seed)
    patient_info = patient_info.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    target_val_patients = int(math.ceil(n_patients * val_frac))
    max_val_patients = int(math.ceil(n_patients * val_max_frac))
    target_val_patients = min(max(1, target_val_patients), n_patients - 1)
    max_val_patients = min(max(target_val_patients, max_val_patients), n_patients - 1)

    val_patients = patient_info.head(target_val_patients)["patient_id"].tolist()
    all_classes = set(patient_info["label"].unique().tolist())

    def current_val_classes():
        return set(patient_info.loc[patient_info["patient_id"].isin(val_patients), "label"].unique().tolist())

    val_classes = current_val_classes()
    if len(val_classes) < 2 and len(all_classes) >= 2:
        missing_classes = sorted(all_classes - val_classes)
        for missing_class in missing_classes:
            candidates = patient_info[
                (~patient_info["patient_id"].isin(val_patients))
                & (patient_info["label"] == missing_class)
            ]["patient_id"].tolist()
            for patient_id in candidates:
                if len(val_patients) >= max_val_patients:
                    break
                val_patients.append(patient_id)
                val_classes = current_val_classes()
                if len(val_classes) >= 2:
                    break

    val_patient_set = set(val_patients)
    val_mask = train_mask & df["patient_id"].isin(val_patient_set)
    df.loc[train_mask, "fold"] = 0
    df.loc[val_mask, "split"] = "val"

    val_rows = int(val_mask.sum())
    train_rows = int((df["split"] == "train").sum())
    val_patient_info = patient_info[patient_info["patient_id"].isin(val_patient_set)]
    val_class_count = val_patient_info["label"].nunique()
    print(
        f"[Split] n_folds=0 train/val split: val_frac={val_frac:.3f}, "
        f"val_max_frac={val_max_frac:.3f}"
    )
    print(
        f"  Train: {train_rows} images, {df.loc[df['split'] == 'train', 'patient_id'].nunique()} patients"
    )
    print(
        f"  Val:   {val_rows} images, {len(val_patient_set)} patients, "
        f"classes={sorted(val_patient_info['label'].unique().tolist())}"
    )
    if val_class_count < 2:
        print(
            "  WARNING: Validation split still has one class after expansion. "
            "Increase kfold0_val_max_frac or add more training data with both classes."
        )

    return df


def create_folds(csv_path, label_col="cancer", n_folds=5, seed=42, output_path=None,
                 overlap_policy="test", split_by_cohort=False, cohort_col="cohort_num",
                 train_cohorts="1-8", test_cohorts="9-10",
                 kfold0_val_frac=0.2, kfold0_val_max_frac=0.5):
    """Create patient-grouped folds and a normalized train/test split CSV."""
    rng = np.random.RandomState(seed)
    overlap_policy = str(overlap_policy or "test").strip().lower()
    overlap_aliases = {
        "raise": "error",
        "strict": "error",
        "train": "train",
        "training": "train",
    }
    overlap_policy = overlap_aliases.get(overlap_policy, overlap_policy)
    allowed_overlap_policies = {"error", "test", "train"}
    if overlap_policy not in allowed_overlap_policies:
        raise ValueError(
            f"Unsupported overlap_policy={overlap_policy!r}. "
            f"Expected one of {sorted(allowed_overlap_policies)}."
        )

    df = read_mammo_csv(csv_path)
    split_by_cohort = parse_bool(split_by_cohort, default=False)
    required_cols = {"patient_id", "image_id", label_col}
    if not split_by_cohort:
        required_cols.add("split")
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing_cols)}")
    if n_folds < 0 or n_folds == 1:
        raise ValueError(f"n_folds must be 0 or >= 2, got {n_folds}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df[label_col] = pd.to_numeric(df[label_col], errors="raise").astype(int)
    if split_by_cohort:
        resolved_cohort_col = resolve_cohort_column(df, cohort_col)
        df[resolved_cohort_col] = pd.to_numeric(df[resolved_cohort_col], errors="raise").astype(int)
        train_cohort_set = parse_cohort_spec(train_cohorts)
        test_cohort_set = parse_cohort_spec(test_cohorts)
        overlap_cohorts = sorted(train_cohort_set & test_cohort_set)
        if overlap_cohorts:
            raise ValueError(f"Train/test cohort specs overlap: {overlap_cohorts}")
        df["split"] = ""
        df.loc[df[resolved_cohort_col].isin(train_cohort_set), "split"] = "train"
        df.loc[df[resolved_cohort_col].isin(test_cohort_set), "split"] = "test"
        unmapped = df["split"] == ""
        if unmapped.any():
            bad_cohorts = sorted(df.loc[unmapped, resolved_cohort_col].dropna().unique().tolist())
            raise ValueError(
                f"{int(unmapped.sum())} row(s) have cohort values outside train/test specs: {bad_cohorts}"
            )
        print(f"[Split] Using cohort split from column {resolved_cohort_col!r}: "
              f"train={sorted(train_cohort_set)}, test={sorted(test_cohort_set)}")
    else:
        df["split"] = normalize_split_values(df["split"])

    bad_labels = sorted(set(df[label_col].dropna().unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"Unsupported label value(s) in {label_col}: {bad_labels}. Expected 0/1.")

    print(f"[Folds] Loaded {len(df)} rows. split counts:\n{df['split'].value_counts()}")

    df["fold"] = -1
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    train_patients = set(df.loc[train_mask, "patient_id"])
    test_patients = set(df.loc[test_mask, "patient_id"])
    overlap = sorted(train_patients & test_patients)
    if overlap:
        message = (
            f"{len(overlap)} patient(s) appear in both training and test split. "
            f"Examples: {overlap[:5]}"
        )
        if overlap_policy == "error":
            raise ValueError(message)
        overlap_mask = df["patient_id"].isin(overlap)
        overlap_split_counts = df.loc[overlap_mask, "split"].value_counts().to_dict()
        print(f"[Folds] WARNING: {message}")
        print(f"[Folds] Overlap row split counts before fix: {overlap_split_counts}")
        print(f"[Folds] Resolving overlap by assigning all overlap patients to split={overlap_policy!r}.")
        df.loc[overlap_mask, "split"] = overlap_policy
        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"
        print(f"[Folds] split counts after overlap fix:\n{df['split'].value_counts()}")

    train_df = df[train_mask].copy()
    if train_df.empty:
        raise ValueError("No rows with split == 'train'; cannot create folds.")

    if n_folds == 0:
        df = assign_kfold0_validation_split(
            df,
            train_mask,
            label_col,
            kfold0_val_frac,
            kfold0_val_max_frac,
            seed,
        )
        csv_path = Path(csv_path)
        if output_path is None:
            output_path = Path.cwd() / f"{csv_path.stem}_folds{csv_path.suffix}"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Folds] Saved single-run split CSV -> {output_path}")
        print(f"  Train (fold=0): {int((df['split'] == 'train').sum())} images, "
              f"{df.loc[df['split'] == 'train', 'patient_id'].nunique()} patients")
        print(f"  Val (fold=0): {int((df['split'] == 'val').sum())} images, "
              f"{df.loc[df['split'] == 'val', 'patient_id'].nunique()} patients")
        print(f"  Test (fold=-1): {int(test_mask.sum())} images, "
              f"{df.loc[test_mask, 'patient_id'].nunique()} patients")
        return output_path

    patient_info = train_df.groupby("patient_id").agg(
        label=(label_col, "max"),
        n_images=("image_id", "count"),
    ).reset_index()
    patient_info["label"] = patient_info["label"].astype(int)

    pos_patients = patient_info[patient_info["label"] == 1]["patient_id"].values
    neg_patients = patient_info[patient_info["label"] == 0]["patient_id"].values

    if len(patient_info) < n_folds:
        raise ValueError(f"n_folds={n_folds} > number of training patients ({len(patient_info)}).")
    if len(pos_patients) < n_folds or len(neg_patients) < n_folds:
        raise ValueError(
            f"Cannot make {n_folds} folds with both classes in every fold: "
            f"positive patients={len(pos_patients)}, negative patients={len(neg_patients)}."
        )

    print(f"[Folds] Training patients: {len(patient_info)} "
          f"(pos={len(pos_patients)}, neg={len(neg_patients)})")
    print(f"[Folds] Training images:   {len(train_df)} "
          f"(pos={(train_df[label_col] == 1).sum()}, neg={(train_df[label_col] == 0).sum()})")

    def assign_folds(fold_patient_lists):
        df["fold"] = -1
        for fold_id, fold_patients in enumerate(fold_patient_lists):
            fold_patient_set = set(fold_patients)
            df.loc[train_mask & df["patient_id"].isin(fold_patient_set), "fold"] = fold_id

    def manual_fold_patient_lists():
        pos = pos_patients.copy()
        neg = neg_patients.copy()
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_folds = np.array_split(pos, n_folds)
        neg_folds = np.array_split(neg, n_folds)
        return [np.concatenate([pos_folds[f], neg_folds[f]]) for f in range(n_folds)]

    def validate_assignment():
        errors = []
        training = df[train_mask].copy()
        if (training["fold"] < 0).any():
            errors.append("Some training rows were not assigned to a fold.")
        if not (df.loc[test_mask, "fold"] == -1).all():
            errors.append("Some test rows were assigned to a training fold.")

        patient_fold_counts = training.groupby("patient_id")["fold"].nunique()
        leaked = patient_fold_counts[patient_fold_counts > 1]
        if len(leaked) > 0:
            errors.append(f"{len(leaked)} training patient(s) appear in multiple folds.")

        patient_folds = training.groupby("patient_id")["fold"].first()
        assigned_info = patient_info.set_index("patient_id").join(patient_folds.rename("fold"))
        for fold_id in range(n_folds):
            fold_info = assigned_info[assigned_info["fold"] == fold_id]
            pos_count = int((fold_info["label"] == 1).sum())
            neg_count = int((fold_info["label"] == 0).sum())
            if pos_count == 0:
                errors.append(f"Fold {fold_id} has zero positive patients.")
            if neg_count == 0:
                errors.append(f"Fold {fold_id} has zero negative patients.")
        return errors

    patient_ids = patient_info["patient_id"].values
    y_patient = patient_info["label"].values.astype(int)
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_patient_lists = [None] * n_folds
        for fold_id, (_, val_patient_pos) in enumerate(
            sgkf.split(patient_ids, y_patient, groups=patient_ids)
        ):
            fold_patient_lists[fold_id] = patient_ids[val_patient_pos]
        assign_folds(fold_patient_lists)
        validation_errors = validate_assignment()
        if validation_errors:
            print("[Folds] StratifiedGroupKFold produced an invalid fold layout; falling back to manual split.")
            for error in validation_errors:
                print(f"  - {error}")
            assign_folds(manual_fold_patient_lists())
    except ValueError as e:
        print(f"[Folds] StratifiedGroupKFold failed ({e}). Falling back to manual split.")
        assign_folds(manual_fold_patient_lists())

    validation_errors = validate_assignment()
    if validation_errors:
        raise ValueError("Invalid fold assignment:\n  - " + "\n  - ".join(validation_errors))

    csv_path = Path(csv_path)
    if output_path is None:
        output_path = Path.cwd() / f"{csv_path.stem}_folds{csv_path.suffix}"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"[Folds] Saved {n_folds}-fold CSV -> {output_path}")

    patient_folds = df[train_mask].groupby("patient_id")["fold"].first()
    assigned_info = patient_info.set_index("patient_id").join(patient_folds.rename("fold"))
    for fold_id in range(n_folds):
        fold_count = int((df["fold"] == fold_id).sum())
        fold_info = assigned_info[assigned_info["fold"] == fold_id]
        patient_count = len(fold_info)
        pos_patient_count = int((fold_info["label"] == 1).sum())
        neg_patient_count = int((fold_info["label"] == 0).sum())
        pos_image_count = int(((df["fold"] == fold_id) & (df[label_col] == 1)).sum())
        print(f"  Fold {fold_id}: {fold_count} images, {patient_count} patients "
              f"(pos_patients={pos_patient_count}, neg_patients={neg_patient_count}), "
              f"cancer_images={pos_image_count}")
    print(f"  Test (fold=-1): {(df['fold'] == -1).sum()} images, "
          f"{df[df['fold'] == -1]['patient_id'].nunique()} patients")
    return output_path


# ==================== EDL 辅助函数 ====================

def labels_to_onehot(labels, num_classes=2, device="cpu"):
    """将标签转换为 one-hot 编码

    Args:
        labels: [batch_size] 浮点标签 (0.0 或 1.0)
        num_classes: 类别数
        device: 设备

    Returns:
        one-hot: [batch_size, num_classes]
    """
    label_long = labels.long()
    onehot = F.one_hot(label_long, num_classes=num_classes).float().to(device)
    return onehot


def compute_fold_class_weights(train_df, label_col, weighted_bce):
    """Return EDL data-loss class weights using the legacy weighted_BCE switch."""
    if not weighted_bce:
        return None
    n_pos = int((train_df[label_col] == 1).sum())
    n_neg = int((train_df[label_col] == 0).sum())
    if n_pos <= 0:
        print("[EDL] No positive samples in this fold train split; using unweighted data loss.")
        return None
    pos_weight = float(n_neg / max(n_pos, 1))
    print(f"[EDL] weighted_BCE=y -> EDL data-loss class weights: "
          f"neg=1.0000, pos={pos_weight:.4f} (n_neg={n_neg}, n_pos={n_pos})")
    return [1.0, pos_weight]


def compute_edl_outputs(logits, evidence_type="softplus", num_classes=2):
    """从 logits 计算 EDL 全部输出

    Args:
        logits: [B, num_classes]
        evidence_type: evidence 激活类型
        num_classes: 类别数

    Returns:
        dict: evidence, alpha, probability, uncertainty
    """
    evidence = get_evidence(logits, evidence_type)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    probability = alpha / S
    uncertainty = num_classes / S

    return {
        "evidence": evidence,
        "alpha": alpha,
        "probability": probability,
        "uncertainty": uncertainty,
    }


def save_loss_curve(history_df, output_path, title):
    """Save a train/valid loss curve as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", linewidth=2, label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], marker="s", linewidth=2, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_all_folds_loss_curve(history_df, output_path, title):
    """Save a multi-panel loss plot for all folds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted(history_df["fold"].unique().tolist())
    if not folds:
        return

    ncols = 2 if len(folds) > 1 else 1
    nrows = math.ceil(len(folds) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, fold in zip(axes_flat, folds):
        fold_df = history_df[history_df["fold"] == fold].sort_values("epoch")
        ax.plot(fold_df["epoch"], fold_df["train_loss"], marker="o", linewidth=2, label="train_loss")
        ax.plot(fold_df["epoch"], fold_df["val_loss"], marker="s", linewidth=2, label="val_loss")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    for ax in axes_flat[len(folds):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_edl_annealing_value(epoch, total_epochs, annealing_step=None, annealing_start_frac=0.0):
    if annealing_step not in (None, ""):
        annealing_step = float(annealing_step)
        return min(1.0, (float(epoch) + 1.0) / max(annealing_step, 1.0))

    annealing_start_epoch = float(annealing_start_frac) * float(total_epochs)
    if float(epoch) < annealing_start_epoch:
        return 0.0

    progress = (float(epoch) - annealing_start_epoch) / max(float(total_epochs) - annealing_start_epoch, 1.0)
    return min(1.0, progress)


def is_edl_annealing_complete(epoch, total_epochs, annealing_step=None, annealing_start_frac=0.0):
    return get_edl_annealing_value(
        epoch,
        total_epochs,
        annealing_step=annealing_step,
        annealing_start_frac=annealing_start_frac,
    ) >= 1.0 - 1e-12


# ==================== Training ====================

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                epoch, total_epochs, args, logger, device):
    """训练一个 epoch"""
    model.train()
    losses = AverageMeter()
    start = time.time()
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    num_classes = getattr(args, "num_classes", 2)
    evidence_type = getattr(args, "evidence_type", "softplus")

    # 更新损失函数的 epoch（用于 KL 退火）
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    for step, data in enumerate(tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs} train]")):
        inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
        labels = data["y"].to(device, non_blocking=True)
        bs = inputs.size(0)

        # 转换为 one-hot
        labels_onehot = labels_to_onehot(labels, num_classes=num_classes, device=device)

        optimizer.zero_grad(set_to_none=True)
        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_inputs = inputs[mb_start:mb_end]
            mb_labels_onehot = labels_onehot[mb_start:mb_end]
            mb_size = mb_inputs.size(0)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(mb_inputs)
                loss = criterion(logits, mb_labels_onehot)
                scaled_loss = loss * (mb_size / bs)

            losses.update(loss.item(), mb_size)
            scaler.scale(scaled_loss).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.print_freq == 0 or step == len(loader) - 1:
            print(f"Epoch [{epoch+1}][{step}/{len(loader)}] "
                  f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                  f"Remain: {timeSince(start, float(step+1)/len(loader))}")

        if step % args.log_freq == 0 or step == len(loader) - 1:
            idx = step + len(loader) * epoch
            logger.add_scalar("train/iter_loss", losses.avg, idx)
            logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], idx)

    return losses.avg


@torch.no_grad()
def valid_epoch(model, loader, criterion, epoch, total_epochs, args, device):
    """验证一个 epoch，返回损失和 EDL 预测结果"""
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_evidence = []
    all_alpha = []
    all_uncertainty = []
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    num_classes = getattr(args, "num_classes", 2)
    evidence_type = getattr(args, "evidence_type", "softplus")

    # 更新损失函数的 epoch
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    for data in tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs} valid]"):
        inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
        labels = data["y"].to(device, non_blocking=True)
        bs = inputs.size(0)

        labels_onehot = labels_to_onehot(labels, num_classes=num_classes, device=device)

        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_inputs = inputs[mb_start:mb_end]
            mb_labels_onehot = labels_onehot[mb_start:mb_end]
            mb_size = mb_inputs.size(0)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(mb_inputs)
                loss = criterion(logits, mb_labels_onehot)

            edl_out = compute_edl_outputs(logits, evidence_type, num_classes)

            losses.update(loss.item(), mb_size)
            all_probs.append(edl_out["probability"].cpu().numpy())
            all_evidence.append(edl_out["evidence"].cpu().numpy())
            all_alpha.append(edl_out["alpha"].cpu().numpy())
            all_uncertainty.append(edl_out["uncertainty"].cpu().numpy())

    if not all_probs:
        raise ValueError("Validation loader produced no batches.")

    predictions = np.concatenate(all_probs)      # [N, num_classes]
    evidences = np.concatenate(all_evidence)      # [N, num_classes]
    alphas = np.concatenate(all_alpha)            # [N, num_classes]
    uncertainties = np.concatenate(all_uncertainty)  # [N, 1]

    return losses.avg, predictions, evidences, alphas, uncertainties


# ==================== Prediction ====================

@torch.no_grad()
def predict_all_edl(model_paths, df_all, img_dir, args, device, threshold=None):
    """使用所有 fold 模型进行 EDL 预测

    Args:
        model_paths: [(fold_idx, ckpt_path), ...]
        df_all: 完整数据 DataFrame
        img_dir: 图片目录
        args: 参数
        device: 设备
        threshold: 二值化阈值

    Returns:
        ensemble_results: dict 包含 ensemble 的 pred_score, pred_label, evidence, uncertainty 等
        per_fold_results: list of dict，每个 fold 的预测结果
    """
    base_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if base_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["model_type"]

    num_classes = getattr(args, "num_classes", 2)
    evidence_type = getattr(args, "evidence_type", "softplus")
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    th = 0.5 if threshold is None else float(threshold)

    all_scores = []
    all_probabilities = []
    all_evidences = []
    all_alphas = []
    all_uncertainties = []
    per_fold_results = []

    for model_idx, model_item in enumerate(model_paths):
        if isinstance(model_item, (tuple, list)):
            fold_idx, ckpt_path = model_item
        else:
            fold_idx, ckpt_path = model_idx, model_item
        print(f"[Predict] Loading fold {fold_idx} model: {ckpt_path}")

        model = MammoEDLModel(args, ckpt=base_ckpt, num_classes=num_classes,
                              evidence_type=evidence_type)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model = model.to(device)
        model.eval()

        dataset = CustomMammoDataset(df_all, img_dir,
                                     label_col=args.label,
                                     transform=None,
                                     mean=args.mean, std=args.std)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, collate_fn=collate_fn)

        fold_probs = []
        fold_evidence = []
        fold_alpha = []
        fold_uncertainty = []

        for data in tqdm(loader, desc=f"[Predict] fold {fold_idx}"):
            inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
            bs = inputs.size(0)
            for mb_start in range(0, bs, micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, bs)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(inputs[mb_start:mb_end])
                edl_out = compute_edl_outputs(logits, evidence_type, num_classes)
                fold_probs.append(edl_out["probability"].cpu().numpy())
                fold_evidence.append(edl_out["evidence"].cpu().numpy())
                fold_alpha.append(edl_out["alpha"].cpu().numpy())
                fold_uncertainty.append(edl_out["uncertainty"].cpu().numpy())

        fold_prob = np.concatenate(fold_probs)           # [N, num_classes]
        fold_evid = np.concatenate(fold_evidence)         # [N, num_classes]
        fold_alph = np.concatenate(fold_alpha)            # [N, num_classes]
        fold_unc = np.concatenate(fold_uncertainty)       # [N, 1]

        # 正类概率 (类别索引 1)
        fold_score = fold_prob[:, 1]
        fold_label = (fold_score >= th).astype(int)

        all_scores.append(fold_score)
        all_probabilities.append(fold_prob)
        all_evidences.append(fold_evid)
        all_alphas.append(fold_alph)
        all_uncertainties.append(fold_unc)

        per_fold_results.append({
            "fold": fold_idx,
            "pred_score": fold_score,
            "pred_label": fold_label,
            "probability": fold_prob,
            "evidence": fold_evid,
            "alpha": fold_alph,
            "uncertainty": fold_unc,
        })

        torch.cuda.empty_cache()

    # Ensemble: 取平均
    pred_score = np.mean(all_scores, axis=0)
    pred_probability = np.mean(all_probabilities, axis=0)
    pred_evidence = np.mean(all_evidences, axis=0)
    pred_alpha = np.mean(all_alphas, axis=0)
    pred_uncertainty = np.mean(all_uncertainties, axis=0)
    pred_label = (pred_score >= th).astype(int)

    print(f"[Predict] Binarization threshold: {th:.4f}")

    ensemble_results = {
        "pred_score": pred_score,
        "pred_label": pred_label,
        "probability": pred_probability,
        "evidence": pred_evidence,
        "alpha": pred_alpha,
        "uncertainty": pred_uncertainty,
    }

    return ensemble_results, per_fold_results


# ==================== CSV 输出 ====================

def save_prediction_csv(df_all, results, output_path, fold_idx=None):
    """保存预测结果 CSV，包含 EDL 相关信息

    Args:
        df_all: 原始数据 DataFrame
        results: dict 包含 pred_score, pred_label, evidence, alpha, uncertainty
        output_path: 输出路径
        fold_idx: fold 编号（None 表示 ensemble）
    """
    df_out = df_all.copy()

    # 预测分数和标签
    df_out["pred_score"] = results["pred_score"]
    df_out["pred_label"] = results["pred_label"]

    # Evidence 值
    num_classes = results["evidence"].shape[1]
    for k in range(num_classes):
        df_out[f"evidence_{k}"] = results["evidence"][:, k]

    # Alpha (Dirichlet 参数)
    for k in range(num_classes):
        df_out[f"alpha_{k}"] = results["alpha"][:, k]

    # Dirichlet predictive probability
    if "probability" in results:
        for k in range(num_classes):
            df_out[f"probability_{k}"] = results["probability"][:, k]

    # Uncertainty
    df_out["uncertainty"] = results["uncertainty"].squeeze()

    # Fold 信息
    if "fold" in df_out.columns:
        # 保留原始 fold 列
        pass
    else:
        df_out["fold"] = -1

    if fold_idx is not None:
        df_out["source_model"] = f"fold{fold_idx}"
    else:
        df_out["source_model"] = "ensemble"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[Save] Predictions saved → {output_path}")
    return df_out


def save_fold_split_csv(df, output_path, fold, n_folds):
    df_out = df.copy()
    if n_folds > 0:
        train_pool_mask = df_out["split"] == "train"
        df_out.loc[train_pool_mask & (df_out["fold"] == fold), "split"] = "val"
        df_out.loc[train_pool_mask & (df_out["fold"] != fold), "split"] = "train"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[Split] Fold {fold} split CSV saved -> {output_path}")
    return output_path


def save_metrics_csv(metrics, output_path, split_name):
    rows = []
    for metric, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            rows.append({"split": split_name, "metric": metric, "value": float(value)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[Final] Metrics saved -> {output_path}")


# ==================== Main ====================

def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(
            description="5-Fold CV fine-tuning with EDL for breast cancer detection",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # 路径参数
        parser.add_argument("--data-dir", default=r"G:\data", type=str)
        parser.add_argument("--img-dir", default="images_png", type=str)
        parser.add_argument("--csv-file", default="train_with_test_data.csv", type=str)
        parser.add_argument("--clip_chk_pt_path",
                            default="./model/Mammo-FM_BatmanlabTrained_CLIP.tar",
                            type=str)
        parser.add_argument("--model-save-dir", default=None, type=str)
        parser.add_argument("--csv-output-dir", default=None, type=str)
        parser.add_argument("--output-dir", default=None, type=str)
        parser.add_argument("--fold-csv", default=None, type=str)
        parser.add_argument("--use-existing-fold-csv", default="n", type=str)
        parser.add_argument("--overlap-policy", default="test",
                            choices=["error", "test", "train", "training"])
        parser.add_argument("--split-by-cohort", default="y", type=str)
        parser.add_argument("--cohort-col", default="cohort_num", type=str)
        parser.add_argument("--train-cohorts", default="1-8", type=str)
        parser.add_argument("--test-cohorts", default="9-10", type=str)

        # 数据参数
        parser.add_argument("--dataset", default="Custom", type=str)
        parser.add_argument("--data_frac", default="1.0", type=str)
        parser.add_argument("--label", default="cancer", type=str)
        parser.add_argument("--arch", default="breast_clip_det_b5_period_n_ft",
                            choices=["breast_clip_det_b5_period_n_lp", "breast_clip_det_b5_period_n_ft"])
        parser.add_argument("--freeze-backbone", default="n", choices=["y", "n"])

        # 训练参数
        parser.add_argument("--n_folds", default=5, type=int)
        parser.add_argument("--kfold0-val-frac", default=0.2, type=float)
        parser.add_argument("--kfold0-val-max-frac", default=0.5, type=float)
        parser.add_argument("--epochs", default=10, type=int)
        parser.add_argument("--early-stop", default=5, type=int)
        parser.add_argument("--batch-size", default=4, type=int)
        parser.add_argument("--micro-batch-size", default=1, type=int)
        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--weight-decay", default=1e-4, type=float)
        parser.add_argument("--weighted_BCE", "--weighted-bce", dest="weighted_BCE", default="y", type=str)
        parser.add_argument("--warmup-epochs", default=1, type=float)
        parser.add_argument("--img-size", nargs="+", default=[1520, 912], type=int)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--num-workers", default=2, type=int)

        # 设备参数
        parser.add_argument("--gpu-id", default=0, type=int)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--apex", default="y", type=str)
        parser.add_argument("--print-freq", default=50, type=int)
        parser.add_argument("--log-freq", default=200, type=int)

        # 数据增强参数
        parser.add_argument("--alpha", default=10, type=float)
        parser.add_argument("--sigma", default=15, type=float)
        parser.add_argument("--p", default=1.0, type=float)

        # 归一化参数
        parser.add_argument("--mean", default=0.3089279, type=float)
        parser.add_argument("--std", default=0.25053555408335154, type=float)

        # EDL 特有参数
        parser.add_argument("--evidence-type", default="softplus", choices=["relu", "exp", "softplus"])
        parser.add_argument("--edl-loss-type", default="log", choices=["log", "digamma", "mse"])
        parser.add_argument("--annealing-coef", default=0.1, type=float)
        parser.add_argument("--edl-kl-weight", default=None, type=float)
        parser.add_argument("--annealing-step", default=None, type=float)
        parser.add_argument("--annealing-start-frac", default=0.0, type=float)

        args = parser.parse_args()

    # ---- 设置 EDL 相关属性 ----
    args.num_classes = 2  # 二分类
    freeze_backbone = getattr(args, "freeze_backbone", False)
    if isinstance(freeze_backbone, str):
        freeze_backbone = freeze_backbone.lower() in {"1", "true", "t", "yes", "y"}
    args.freeze_backbone = bool(freeze_backbone)
    args.evidence_type = str(getattr(args, "evidence_type", "softplus")).lower()
    if not hasattr(args, "edl_loss_type"):
        args.edl_loss_type = getattr(args, "edl_loss_type", "log")
    if not hasattr(args, "annealing_coef"):
        args.annealing_coef = getattr(args, "annealing_coef", 0.1)
    if getattr(args, "edl_kl_weight", None) is not None:
        args.annealing_coef = float(args.edl_kl_weight)
    else:
        args.edl_kl_weight = float(args.annealing_coef)
    if not hasattr(args, "annealing_start_frac"):
        args.annealing_start_frac = getattr(args, "annealing_start_frac", 0.0)
    if not hasattr(args, "annealing_step"):
        args.annealing_step = getattr(args, "annealing_step", None)
    if args.annealing_step in ("",):
        args.annealing_step = None
    if args.annealing_step is not None:
        args.annealing_step = float(args.annealing_step)
        if args.annealing_step <= 0:
            raise ValueError(f"annealing_step must be > 0, got {args.annealing_step}")
    if not hasattr(args, "weighted_BCE"):
        args.weighted_BCE = getattr(args, "weighted_bce", "y")
    args.weighted_bce = str(args.weighted_BCE).lower() in {"1", "true", "t", "yes", "y"}
    args.use_existing_fold_csv = parse_bool(
        getattr(args, "use_existing_fold_csv", False),
        default=False,
    )
    args.n_folds = int(args.n_folds)
    if args.n_folds < 0 or args.n_folds == 1:
        raise ValueError(f"n_folds must be 0 or >= 2, got {args.n_folds}")
    args.kfold0_val_frac = normalize_fraction_arg(
        getattr(args, "kfold0_val_frac", 0.2),
        "kfold0_val_frac",
    )
    args.kfold0_val_max_frac = normalize_fraction_arg(
        getattr(args, "kfold0_val_max_frac", 0.5),
        "kfold0_val_max_frac",
    )
    if args.kfold0_val_max_frac < args.kfold0_val_frac:
        raise ValueError(
            f"kfold0_val_max_frac ({args.kfold0_val_max_frac:.4f}) must be >= "
            f"kfold0_val_frac ({args.kfold0_val_frac:.4f})."
        )
    args.split_by_cohort = parse_bool(getattr(args, "split_by_cohort", True), default=True)
    args.cohort_col = str(getattr(args, "cohort_col", "cohort_num"))
    args.train_cohorts = str(getattr(args, "train_cohorts", "1-8"))
    args.test_cohorts = str(getattr(args, "test_cohorts", "9-10"))
    overlap_policy = str(getattr(args, "overlap_policy", "test")).strip().lower()
    overlap_policy = {"raise": "error", "strict": "error", "training": "train"}.get(
        overlap_policy, overlap_policy
    )
    if overlap_policy not in {"error", "test", "train"}:
        raise ValueError(
            f"Unsupported overlap_policy={overlap_policy!r}. Expected error, test, or train."
        )
    args.overlap_policy = overlap_policy

    # ---- GPU 设置 ----
    gpu_id = int(getattr(args, "gpu_id", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    data_dir = resolve_project_path(args.data_dir)
    args.data_dir = data_dir
    if Path(args.img_dir).is_absolute() or os.path.isabs(str(args.img_dir)):
        args.img_dir = Path(args.img_dir)
    else:
        args.img_dir = data_dir / args.img_dir

    if Path(args.csv_file).is_absolute() or os.path.isabs(str(args.csv_file)):
        args.data_csv = Path(args.csv_file)
    else:
        args.data_csv = data_dir / args.csv_file
    args.clip_chk_pt_path = resolve_project_path(args.clip_chk_pt_path)

    output_arg = getattr(args, "output_dir", None)
    model_save_arg = getattr(args, "model_save_dir", None)
    csv_output_arg = getattr(args, "csv_output_dir", None)
    if model_save_arg in (None, "") and output_arg not in (None, ""):
        model_save_dir = resolve_project_path(Path(output_arg) / "checkpoints")
    else:
        model_save_dir = resolve_project_path(model_save_arg, "./best_model")
    if csv_output_arg in (None, "") and output_arg not in (None, ""):
        csv_output_dir = resolve_project_path(output_arg)
    else:
        csv_output_dir = resolve_project_path(csv_output_arg, "./output")
    ckpt_dir = model_save_dir
    log_dir = model_save_dir / "tensorboard"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.model_save_dir = model_save_dir
    args.csv_output_dir = csv_output_dir
    args.output_dir = csv_output_dir
    output_dir = csv_output_dir

    args.apex = str(args.apex).lower() == "y"
    args.data_frac = float(args.data_frac)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    print("=" * 60)
    print(f"  Evidential Deep Learning (EDL) Fine-tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model Save Dir: {model_save_dir}")
    print(f"CSV Output Dir: {csv_output_dir}")
    print(f"Data CSV: {args.data_csv}")
    print(f"Image dir: {args.img_dir}")
    print(f"Checkpoint: {args.clip_chk_pt_path}")
    print(f"Arch: {args.arch}  |  Folds: {args.n_folds}  |  EarlyStop: {args.early_stop}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Epochs(max): {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Dataset: {args.dataset}  |  Label: {args.label}  |  Data frac: {args.data_frac}")
    print(f"AMP: {args.apex}")
    print(f"EDL Evidence Type: {args.evidence_type}")
    print(f"EDL Loss Type: {args.edl_loss_type}")
    print(f"EDL KL Weight(lambda): {args.edl_kl_weight}")
    print(f"EDL Annealing Step: {args.annealing_step}")
    print(f"EDL Annealing Start Frac: {args.annealing_start_frac}")
    print(f"Weighted BCE/Data Loss: {args.weighted_bce}")
    print(f"Overlap Policy: {args.overlap_policy}")
    print(f"Split By Cohort: {args.split_by_cohort}")
    print(f"Use Existing Fold CSV: {args.use_existing_fold_csv}")
    if args.n_folds == 0:
        print(f"KFold0 Val Frac: {args.kfold0_val_frac}")
        print(f"KFold0 Val Max Frac: {args.kfold0_val_max_frac}")
    if args.split_by_cohort:
        print(f"Cohort Column: {args.cohort_col}  |  Train: {args.train_cohorts}  |  Test: {args.test_cohorts}")
    print(f"Num Classes: {args.num_classes}")
    print("=" * 60)

    # ---- 创建 Folds ----
    if getattr(args, "fold_csv", None) not in (None, ""):
        fold_csv_path = Path(args.fold_csv)
        if not fold_csv_path.is_absolute() and not os.path.isabs(str(fold_csv_path)):
            fold_csv_path = csv_output_dir / fold_csv_path
    else:
        fold_csv_path = csv_output_dir / f"{Path(args.data_csv).stem}_folds.csv"

    if args.use_existing_fold_csv:
        if getattr(args, "fold_csv", None) in (None, ""):
            raise ValueError("use_existing_fold_csv=y requires fold_csv to point to an existing folds CSV.")
        if not fold_csv_path.exists():
            raise FileNotFoundError(f"Requested existing folds CSV does not exist: {fold_csv_path}")
        folds_csv = fold_csv_path
        print(f"[Folds] Using existing fold CSV without regenerating -> {folds_csv}")
    else:
        folds_csv = create_folds(args.data_csv, label_col=args.label,
                                 n_folds=args.n_folds, seed=args.seed,
                                 output_path=fold_csv_path,
                                 overlap_policy=args.overlap_policy,
                                 split_by_cohort=args.split_by_cohort,
                                 cohort_col=args.cohort_col,
                                 train_cohorts=args.train_cohorts,
                                 test_cohorts=args.test_cohorts,
                                 kfold0_val_frac=args.kfold0_val_frac,
                                 kfold0_val_max_frac=args.kfold0_val_max_frac)
    df = read_mammo_csv(folds_csv)

    # ---- 加载基础 checkpoint ----
    base_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if base_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["model_type"]

    # ==================== 训练阶段 ====================
    fold_model_paths = []
    oof_parts = []
    all_fold_histories = []
    data_stem = Path(args.data_csv).stem
    fold_ids = [0] if args.n_folds == 0 else list(range(args.n_folds))

    for fold in fold_ids:
        total_runs = 1 if args.n_folds == 0 else args.n_folds
        eval_split_name = "val"
        print(f"\n{'=' * 60}")
        print(f"  Fold {fold} / {total_runs}  |  epoch eval split: {eval_split_name}")
        print(f"{'=' * 60}")

        seed_all(args.seed)

        train_pool_mask = df["split"] == "train"
        if args.n_folds == 0:
            train_df = df[train_pool_mask].reset_index(drop=True)
            valid_df = df[df["split"] == "val"].reset_index(drop=True)
        else:
            train_df = df[train_pool_mask & (df["fold"] != fold)].reset_index(drop=True)
            valid_df = df[train_pool_mask & (df["fold"] == fold)].reset_index(drop=True)
        save_fold_split_csv(df, output_dir / f"{data_stem}_fold{fold}_splits.csv", fold, args.n_folds)

        if args.data_frac < 1.0:
            train_df = train_df.sample(frac=args.data_frac, random_state=args.seed).reset_index(drop=True)

        print(f"Train: {len(train_df)}  |  {eval_split_name.title()}: {len(valid_df)}")
        print(f"  Train cancer%: {train_df[args.label].mean()*100:.1f}%")
        print(f"  {eval_split_name.title()} cancer%: {valid_df[args.label].mean()*100:.1f}%")

        n_valid_classes = valid_df[args.label].nunique()
        if n_valid_classes < 2:
            print(f"  WARNING: {eval_split_name.title()} set for fold {fold} has only {n_valid_classes} class(es).")

        train_transform = get_train_transform(tuple(args.img_size), args.alpha, args.sigma, args.p)
        val_transform = get_val_transform(tuple(args.img_size))

        train_ds = CustomMammoDataset(train_df, args.img_dir, label_col=args.label,
                                      transform=train_transform, mean=args.mean, std=args.std)
        valid_ds = CustomMammoDataset(valid_df, args.img_dir, label_col=args.label,
                                      transform=val_transform, mean=args.mean, std=args.std)

        if len(train_ds) == 0 or len(valid_ds) == 0:
            print(f"  WARNING: Fold {fold} has empty train or valid data; skipping.")
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  drop_last=False, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True,
                                  drop_last=False, collate_fn=collate_fn)

        if len(train_loader) == 0 or len(valid_loader) == 0:
            print(f"  WARNING: Fold {fold} produced no train or valid batches; skipping.")
            continue

        # ---- 创建 EDL 模型 ----
        model = MammoEDLModel(args, ckpt=base_ckpt, num_classes=args.num_classes,
                              evidence_type=args.evidence_type)
        model = model.to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found. Check freeze_backbone/model setup.")
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        backbone_count = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
        head_count = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_count:,} / {total_count:,}")
        print(f"  Trainable backbone params: {backbone_count:,}")
        print(f"  Trainable EDL head params: {head_count:,}")

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = len(train_loader) if args.warmup_epochs >= 1 else int(args.warmup_epochs * len(train_loader))
        lr_config = {
            "total_epochs": args.epochs,
            "warmup_steps": warmup_steps,
            "total_steps": len(train_loader) * args.epochs,
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex)

        # ---- EDL 损失函数 ----
        class_weights = compute_fold_class_weights(train_df, args.label, args.weighted_bce)
        criterion = EDLLogLossWithAnnealing(
            num_classes=args.num_classes,
            evidence_type=args.evidence_type,
            total_epochs=args.epochs,
            annealing_start_frac=args.annealing_start_frac,
            annealing_coef=args.annealing_coef,
            annealing_step=args.annealing_step,
            loss_type=args.edl_loss_type,
            class_weights=class_weights,
        )

        logger = SummaryWriter(log_dir / f"fold{fold}")
        early_stopper = EarlyStopping(patience=args.early_stop) if args.early_stop > 0 else None
        best_auroc = -float("inf")
        best_model_path = ckpt_dir / f"best_fold{fold}_seed{args.seed}_edl.pth"
        saved_best = False
        best_predictions = None
        best_evidences = None
        best_alphas = None
        best_uncertainties = None
        best_metrics = None
        last_predictions = None
        last_evidences = None
        last_alphas = None
        last_uncertainties = None
        last_metrics = None
        last_epoch = -1
        fold_history_rows = []

        try:
            for epoch in range(args.epochs):
                last_epoch = epoch
                start = time.time()

                # 训练
                train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                         scheduler, scaler, epoch, args.epochs,
                                         args, logger, device)

                # 验证
                val_loss, predictions, evidences, alphas, uncertainties = valid_epoch(
                    model, valid_loader, criterion, epoch, args.epochs, args, device
                )
                last_predictions = predictions.copy()
                last_evidences = evidences.copy()
                last_alphas = alphas.copy()
                last_uncertainties = uncertainties.copy()

                # 计算指标（使用正类概率）
                pred_score = predictions[:, 1]  # 正类概率
                valid_df_fold = valid_df.copy()
                valid_df_fold["prediction"] = pred_score
                valid_agg = valid_df_fold.groupby("patient_id").agg({
                    args.label: "max",
                    "prediction": "max",
                })
                metrics = all_classification_metrics(valid_agg[args.label].values,
                                                     valid_agg["prediction"].values)
                last_metrics = metrics
                elapsed = time.time() - start
                mean_uncertainty = float(np.mean(uncertainties))
                annealing_value = get_edl_annealing_value(
                    epoch,
                    args.epochs,
                    annealing_step=args.annealing_step,
                    annealing_start_frac=args.annealing_start_frac,
                )
                annealing_complete = annealing_value >= 1.0 - 1e-12
                is_best_epoch = metrics["AUROC"] > best_auroc
                fold_history_rows.append({
                    "fold": fold,
                    "epoch": epoch + 1,
                    "eval_split": eval_split_name,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "eval_loss": float(val_loss),
                    "auroc": float(metrics["AUROC"]),
                    "auprc": float(metrics["AUPRC"]),
                    "bacc": float(metrics["bACC"]),
                    "mean_uncertainty": mean_uncertainty,
                    "edl_annealing_value": float(annealing_value),
                    "annealing_complete": int(annealing_complete),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_sec": float(elapsed),
                    "is_best": int(is_best_epoch),
                })

                print(f"Fold {fold} Epoch {epoch+1} - train_loss: {train_loss:.4f}  "
                      f"{eval_split_name}_loss: {val_loss:.4f}  AUROC: {metrics['AUROC']:.4f}  "
                      f"AUPRC: {metrics['AUPRC']:.4f}  bACC: {metrics['bACC']:.4f}  time: {elapsed:.0f}s")

                logger.add_scalar(f"{eval_split_name}/AUROC", metrics["AUROC"], epoch + 1)
                logger.add_scalar(f"{eval_split_name}/AUPRC", metrics["AUPRC"], epoch + 1)
                logger.add_scalar(f"{eval_split_name}/bACC", metrics["bACC"], epoch + 1)
                logger.add_scalar(f"{eval_split_name}/loss", val_loss, epoch + 1)
                logger.add_scalar(f"{eval_split_name}/mean_uncertainty", mean_uncertainty, epoch + 1)
                logger.add_scalar("train/edl_annealing_value", annealing_value, epoch + 1)

                if is_best_epoch:
                    best_auroc = metrics["AUROC"]
                    best_predictions = predictions.copy()
                    best_evidences = evidences.copy()
                    best_alphas = alphas.copy()
                    best_uncertainties = uncertainties.copy()
                    best_metrics = metrics
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "auroc": best_auroc,
                        "metrics": metrics,
                    }, best_model_path)
                    saved_best = True
                    print(f"  -> Saved best (AUROC={best_auroc:.4f})")

                if early_stopper is not None:
                    if annealing_complete:
                        if early_stopper(metrics["AUROC"]):
                            print(f"  -> Early stopping at epoch {epoch+1} (best AUROC={early_stopper.best_score:.4f})")
                            break
                    else:
                        print(
                            f"  -> Annealing in progress ({annealing_value:.3f}); "
                            "early stopping will start after annealing completes."
                        )

            if not saved_best:
                print(f"  WARNING: No best model was saved for fold {fold}. Saving last epoch as fallback.")
                best_predictions = last_predictions
                best_evidences = last_evidences
                best_alphas = last_alphas
                best_uncertainties = last_uncertainties
                best_metrics = last_metrics
                torch.save({
                    "epoch": last_epoch,
                    "model": model.state_dict(),
                    "auroc": best_auroc,
                    "metrics": best_metrics,
                }, best_model_path)

        except Exception as e:
            print(f"  ERROR: Fold {fold} training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            if not saved_best:
                try:
                    torch.save({
                        "epoch": -1,
                        "model": model.state_dict(),
                        "auroc": -1.0,
                        "metrics": {},
                    }, best_model_path)
                    print(f"  Saved emergency checkpoint for fold {fold}")
                except Exception as e2:
                    print(f"  Could not save emergency checkpoint: {e2}")

        if fold_history_rows:
            history_df = pd.DataFrame(fold_history_rows)
            history_csv = output_dir / f"{data_stem}_fold{fold}_loss_history.csv"
            history_png = output_dir / f"{data_stem}_fold{fold}_loss_curve.png"
            history_df.to_csv(history_csv, index=False)
            save_loss_curve(history_df, history_png, title=f"Fold {fold} Loss Curve")
            all_fold_histories.append(history_df)
            print(f"  Saved fold {fold} loss history -> {history_csv}")
            print(f"  Saved fold {fold} loss curve   -> {history_png}")

        if best_model_path.exists():
            fold_model_paths.append((fold, best_model_path))
            if args.n_folds > 0 and best_predictions is not None:
                oof_part = valid_df.copy()
                oof_part["oof_pred_score"] = best_predictions[:, 1]
                for k in range(args.num_classes):
                    oof_part[f"oof_probability_{k}"] = best_predictions[:, k]
                    oof_part[f"oof_evidence_{k}"] = best_evidences[:, k]
                    oof_part[f"oof_alpha_{k}"] = best_alphas[:, k]
                oof_part["oof_uncertainty"] = best_uncertainties.squeeze()
                oof_part["oof_fold"] = fold
                oof_parts.append(oof_part)
        else:
            print(f"  WARNING: No checkpoint for fold {fold}, skipping in ensemble.")

        logger.close()
        del model, optimizer, scheduler, scaler, criterion
        torch.cuda.empty_cache()
        gc.collect()

    # ---- 保存 OOF 预测 ----
    if all_fold_histories:
        all_history_df = pd.concat(all_fold_histories, ignore_index=True)
        all_history_csv = output_dir / f"{data_stem}_all_folds_loss_history.csv"
        all_history_png = output_dir / f"{data_stem}_all_folds_loss_curve.png"
        all_history_df.to_csv(all_history_csv, index=False)
        save_all_folds_loss_curve(all_history_df, all_history_png, title="All Folds Loss Curves")
        print(f"\n[Loss] All folds history saved -> {all_history_csv}")
        print(f"[Loss] All folds curves saved  -> {all_history_png}")

    if oof_parts:
        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_csv = output_dir / f"{Path(args.data_csv).stem}_oof_edl_predictions.csv"
        oof_df.to_csv(oof_csv, index=False)
        print(f"\n[OOF] Out-of-fold predictions saved → {oof_csv}")

    # ==================== 自动测试阶段 ====================
    print(f"\n{'=' * 60}")
    print("  Ensemble prediction on ALL data (EDL)")
    print(f"{'=' * 60}")

    if len(fold_model_paths) == 0:
        print("[ERROR] No fold models were saved! Cannot run ensemble prediction.")
        return

    print(f"[Ensemble] Using {len(fold_model_paths)} fold model(s):")
    for fold_id, path in fold_model_paths:
        print(f"  Fold {fold_id}: {path}")

    df_all = read_mammo_csv(folds_csv)
    threshold = 0.5
    print("[Predict] Using fixed threshold=0.5 for pred_label.")

    ensemble_results, per_fold_results = predict_all_edl(
        fold_model_paths, df_all, args.img_dir, args, device, threshold=threshold
    )

    prediction_stem = Path(args.data_csv).stem

    # ---- 保存每个 fold 的预测 CSV ----
    for fold_result in per_fold_results:
        fold_id = fold_result["fold"]
        fold_csv = output_dir / f"{prediction_stem}_predictions_edl_fold{fold_id}.csv"
        save_prediction_csv(df_all, fold_result, fold_csv, fold_idx=fold_id)
        print(f"[Final] Fold {fold_id} EDL predictions saved -> {fold_csv}")

    # ---- 保存 ensemble 预测 CSV ----
    ensemble_csv = output_dir / f"{prediction_stem}_predictions_edl_ensemble.csv"
    ensemble_df = save_prediction_csv(df_all, ensemble_results, ensemble_csv, fold_idx=None)

    print(f"\n[Final] Ensemble EDL predictions saved → {ensemble_csv}")
    print(f"[Final] Columns: {list(ensemble_df.columns)}")
    print(f"[Final] pred_score stats: mean={ensemble_results['pred_score'].mean():.4f} "
          f"std={ensemble_results['pred_score'].std():.4f} "
          f"min={ensemble_results['pred_score'].min():.4f} "
          f"max={ensemble_results['pred_score'].max():.4f}")
    print(f"[Final] mean uncertainty: {ensemble_results['uncertainty'].mean():.4f} "
          f"std: {ensemble_results['uncertainty'].std():.4f}")
    print(f"[Final] pred_label distribution:\n{ensemble_df['pred_label'].value_counts()}")

    # ---- 测试集指标 ----
    if args.label in ensemble_df.columns:
        test_mask = ensemble_df["split"] == "test"
        test_df = ensemble_df[test_mask]
        if len(test_df) > 0:
            metrics = all_classification_metrics(test_df[args.label].values, test_df["pred_score"].values)
            print(f"\n[Final] Test set metrics:")
            print(f"  AUROC: {metrics['AUROC']:.4f}")
            print(f"  AUPRC: {metrics['AUPRC']:.4f}")
            print(f"  bACC:  {metrics['bACC']:.4f}")
            print(f"  Mean uncertainty (test): {test_df['uncertainty'].mean():.4f}")
            metrics_csv = output_dir / f"{prediction_stem}_test_metrics_edl.csv"
            save_metrics_csv(metrics, metrics_csv, split_name="test")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
