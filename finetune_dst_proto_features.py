"""
DST-Prototype training on pre-extracted Mammo-FM embeddings.

Expected embedding directory:
  embeddings.npy
  metadata.csv
  manifest.json  (optional)
"""

import argparse
import gc
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dst_proto_model import DSTPrototypeModel, PrototypeDSTNLLLoss

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DTYPES = {
    "patient_id": str,
    "image_id": str,
    "split": str,
}


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = int(patience)
        self.best_score = -float("inf")
        self.counter = 0

    def __call__(self, score):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def make_summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir)
    except Exception as exc:
        print(f"[TensorBoard] disabled: {exc}")
        return NullSummaryWriter()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_since(start, percent):
    elapsed = time.time() - start
    if percent <= 0:
        return "--"
    total = elapsed / percent
    remain = total - elapsed
    return f"{remain / 60:.1f}m"


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


def resolve_project_path(path_value, default_value=None, base_dir=PROJECT_ROOT):
    if path_value in (None, ""):
        path_value = default_value
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def read_mammo_csv(csv_path):
    return pd.read_csv(csv_path, dtype=CSV_DTYPES)


def normalize_fraction_arg(value, name):
    frac = float(value)
    if frac > 1.0:
        frac /= 100.0
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")
    return frac


def parse_cohort_spec(value):
    tokens = []
    if isinstance(value, (list, tuple, set)):
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
        raise ValueError(f"Unsupported split value(s): {bad}.")
    return normalized


def assign_kfold0_validation_split(df, train_mask, label_col, val_frac, val_max_frac, seed):
    train_df = df.loc[train_mask].copy()
    patient_info = train_df.groupby("patient_id").agg(
        label=(label_col, "max"),
        n_images=("image_id", "count"),
    ).reset_index()
    patient_info = patient_info.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_patients = len(patient_info)
    if n_patients < 2:
        raise ValueError("Need at least 2 training patients for a validation split.")

    target_count = min(max(1, int(math.ceil(n_patients * val_frac))), n_patients - 1)
    max_count = min(max(target_count, int(math.ceil(n_patients * val_max_frac))), n_patients - 1)
    val_patients = patient_info.head(target_count)["patient_id"].tolist()
    all_classes = set(patient_info["label"].unique().tolist())

    def val_classes():
        return set(patient_info.loc[patient_info["patient_id"].isin(val_patients), "label"].unique().tolist())

    if len(val_classes()) < 2 and len(all_classes) >= 2:
        missing_classes = sorted(all_classes - val_classes())
        for missing_class in missing_classes:
            candidates = patient_info[
                (~patient_info["patient_id"].isin(val_patients))
                & (patient_info["label"] == missing_class)
            ]["patient_id"].tolist()
            for patient_id in candidates:
                if len(val_patients) >= max_count:
                    break
                val_patients.append(patient_id)
                if len(val_classes()) >= 2:
                    break

    val_patient_set = set(val_patients)
    df.loc[train_mask, "fold"] = 0
    df.loc[train_mask & df["patient_id"].isin(val_patient_set), "split"] = "val"
    return df


def create_folds(csv_path, label_col, n_folds, seed, output_path, overlap_policy,
                 split_by_cohort, cohort_col, train_cohorts, test_cohorts,
                 kfold0_val_frac, kfold0_val_max_frac):
    df = read_mammo_csv(csv_path)
    required_cols = {"patient_id", "image_id", label_col}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {missing_cols}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df[label_col] = pd.to_numeric(df[label_col], errors="raise").astype(int)
    split_by_cohort = parse_bool(split_by_cohort, default=True)

    if split_by_cohort:
        resolved_cohort_col = resolve_cohort_column(df, cohort_col)
        train_set = parse_cohort_spec(train_cohorts)
        test_set = parse_cohort_spec(test_cohorts)
        overlap = sorted(train_set & test_set)
        if overlap:
            raise ValueError(f"Train/test cohort specs overlap: {overlap}")
        df[resolved_cohort_col] = pd.to_numeric(df[resolved_cohort_col], errors="raise").astype(int)
        df["split"] = ""
        df.loc[df[resolved_cohort_col].isin(train_set), "split"] = "train"
        df.loc[df[resolved_cohort_col].isin(test_set), "split"] = "test"
        if (df["split"] == "").any():
            bad = sorted(df.loc[df["split"] == "", resolved_cohort_col].dropna().unique().tolist())
            raise ValueError(f"Rows have cohort values outside train/test specs: {bad}")
        print(f"[Split] Using cohort split from {resolved_cohort_col!r}.")
    else:
        if "split" not in df.columns:
            raise ValueError("CSV must contain split when split_by_cohort=n.")
        df["split"] = normalize_split_values(df["split"])

    overlap_policy = {"training": "train"}.get(str(overlap_policy).strip().lower(), str(overlap_policy).strip().lower())
    if overlap_policy not in {"error", "test", "train"}:
        raise ValueError(f"Unsupported overlap_policy={overlap_policy!r}.")

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    overlap_patients = sorted(set(df.loc[train_mask, "patient_id"]) & set(df.loc[test_mask, "patient_id"]))
    if overlap_patients:
        if overlap_policy == "error":
            raise ValueError(f"{len(overlap_patients)} patients appear in both train and test.")
        print(f"[Folds] Resolving {len(overlap_patients)} overlap patients as split={overlap_policy!r}.")
        df.loc[df["patient_id"].isin(overlap_patients), "split"] = overlap_policy

    df["fold"] = -1
    train_mask = df["split"] == "train"
    if n_folds == 0:
        df = assign_kfold0_validation_split(
            df,
            train_mask,
            label_col,
            normalize_fraction_arg(kfold0_val_frac, "kfold0_val_frac"),
            normalize_fraction_arg(kfold0_val_max_frac, "kfold0_val_max_frac"),
            seed,
        )
    else:
        train_df = df.loc[train_mask].copy()
        patient_info = train_df.groupby("patient_id").agg(label=(label_col, "max")).reset_index()
        if len(patient_info) < n_folds:
            raise ValueError(f"n_folds={n_folds} > train patients={len(patient_info)}.")
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        patient_ids = patient_info["patient_id"].values
        y_patient = patient_info["label"].values.astype(int)
        for fold_id, (_, val_pos) in enumerate(sgkf.split(patient_ids, y_patient, groups=patient_ids)):
            val_patients = set(patient_ids[val_pos])
            df.loc[train_mask & df["patient_id"].isin(val_patients), "fold"] = fold_id

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Folds] Saved fold CSV -> {output_path}")
    print(df["split"].value_counts())
    return output_path


class EmbeddingFeatureDataset(Dataset):
    def __init__(self, df, embeddings, label_col):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.label_col = label_col
        if "embedding_row" not in self.df.columns:
            self.df["embedding_row"] = np.arange(len(self.df), dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embedding_row = int(row["embedding_row"])
        x = np.asarray(self.embeddings[embedding_row], dtype=np.float32)
        y = int(row[self.label_col])
        return {"x": x, "y": y, "embedding_row": embedding_row}


def embedding_collate_fn(batch):
    return {
        "x": torch.from_numpy(np.stack([item["x"] for item in batch], axis=0)).float(),
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "embedding_row": [item["embedding_row"] for item in batch],
    }


def load_embedding_cache(embedding_dir, embeddings_file, metadata_file):
    embedding_dir = Path(embedding_dir)
    embeddings_path = embedding_dir / embeddings_file
    metadata_path = embedding_dir / metadata_file
    manifest_path = embedding_dir / "manifest.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    metadata = read_mammo_csv(metadata_path)
    if "embedding_row" not in metadata.columns:
        metadata.insert(0, "embedding_row", np.arange(len(metadata), dtype=np.int64))
    metadata["embedding_row"] = pd.to_numeric(metadata["embedding_row"], errors="raise").astype(int)
    if len(metadata) != int(embeddings.shape[0]):
        raise ValueError(
            f"metadata rows ({len(metadata)}) != embeddings rows ({embeddings.shape[0]})."
        )
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return embeddings, metadata, manifest, embeddings_path, metadata_path


def labels_to_onehot(labels, num_classes=2, device="cpu"):
    return F.one_hot(labels.long(), num_classes=num_classes).float().to(device)


def compute_class_weights(train_df, label_col, weighted):
    if not weighted:
        return None
    n_pos = int((train_df[label_col] == 1).sum())
    n_neg = int((train_df[label_col] == 0).sum())
    if n_pos <= 0:
        print("[DST] No positive samples in train split; using unweighted NLL.")
        return None
    pos_weight = float(n_neg / max(n_pos, 1))
    print(f"[DST] class weights: neg=1.0000 pos={pos_weight:.4f}")
    return torch.tensor([1.0, pos_weight], dtype=torch.float32)


def all_classification_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auroc = 0.5
        auprc = float(y_true.mean()) if len(y_true) else 0.0
    else:
        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
    bacc = float(balanced_accuracy_score(y_true, pred))
    return {"AUROC": auroc, "AUPRC": auprc, "bACC": bacc}


class PrototypeRegularizationLoss(torch.nn.Module):
    def __init__(self, attract_weight=0.1, separation_weight=0.1,
                 diversity_weight=0.01, margin=1.0, balance_classes=True):
        super().__init__()
        self.attract_weight = float(attract_weight)
        self.separation_weight = float(separation_weight)
        self.diversity_weight = float(diversity_weight)
        self.margin = float(margin)
        self.balance_classes = bool(balance_classes)

    def _class_balanced_mean(self, values, target_indices, num_classes):
        if values.numel() == 0:
            return values.new_zeros(())
        if not self.balance_classes:
            return values.mean()
        class_means = []
        for class_idx in range(int(num_classes)):
            mask = target_indices == class_idx
            if mask.any():
                class_means.append(values[mask].mean())
        if not class_means:
            return values.new_zeros(())
        return torch.stack(class_means).mean()

    def _diversity_loss(self, proto_head):
        prototypes = proto_head.prototypes
        if getattr(proto_head, "normalize_embeddings", False):
            prototypes = F.normalize(prototypes, p=2, dim=-1, eps=1e-12)
        losses = []
        for class_idx in range(prototypes.size(0)):
            class_prototypes = prototypes[class_idx]
            if class_prototypes.size(0) <= 1:
                continue
            pairwise_dist = torch.cdist(class_prototypes, class_prototypes, p=2)
            tri = torch.triu_indices(
                class_prototypes.size(0),
                class_prototypes.size(0),
                offset=1,
                device=pairwise_dist.device,
            )
            pairwise_dist = pairwise_dist[tri[0], tri[1]]
            if pairwise_dist.numel() > 0:
                losses.append(torch.exp(-pairwise_dist).mean())
        if not losses:
            return prototypes.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, model_outputs, proto_head, target_indices):
        proto_distances = model_outputs["prototype_distances"]
        sample_indices = torch.arange(proto_distances.size(0), device=proto_distances.device)
        same_class_dist = proto_distances[sample_indices, target_indices, :].min(dim=1).values
        other_class_dist = proto_distances[sample_indices, 1 - target_indices, :].min(dim=1).values
        attract_loss = self._class_balanced_mean(same_class_dist, target_indices, proto_distances.size(1))
        separation_loss = self._class_balanced_mean(
            F.relu(self.margin - other_class_dist),
            target_indices,
            proto_distances.size(1),
        )
        diversity_loss = self._diversity_loss(proto_head)
        total_loss = (
            self.attract_weight * attract_loss
            + self.separation_weight * separation_loss
            + self.diversity_weight * diversity_loss
        )
        return {
            "total_loss": total_loss,
            "raw_total_loss": total_loss,
            "attract_loss": attract_loss,
            "separation_loss": separation_loss,
            "diversity_loss": diversity_loss,
        }


def make_scheduler(optimizer, total_steps, warmup_steps):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_metrics_csv(metrics, output_path, split_name):
    rows = []
    for metric, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            rows.append({"split": split_name, "metric": metric, "value": float(value)})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[Metrics] Saved -> {output_path}")


def save_fold_split_csv(df, output_path, fold, n_folds):
    df_out = df.copy()
    if n_folds > 0:
        train_pool_mask = df_out["split"] == "train"
        df_out.loc[train_pool_mask & (df_out["fold"] == fold), "split"] = "val"
        df_out.loc[train_pool_mask & (df_out["fold"] != fold), "split"] = "train"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[Split] Fold split CSV saved -> {output_path}")


def save_loss_curve(history_df, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] skipped: {exc}")
        return
    fig, ax = plt.subplots(figsize=(13.66, 7.68))
    ax.plot(history_df["epoch"], history_df["train_loss"], color="#1f77b4", linewidth=2.5, label="train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], color="#d62728", linewidth=2.5, label="val loss")
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.legend(loc="upper right", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def save_component_curve(history_df, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] skipped: {exc}")
        return
    columns = [
        "train_total_loss",
        "train_dst_loss",
        "train_proto_reg_loss",
        "train_proto_reg_loss_raw",
        "train_proto_attract_loss",
        "train_proto_separation_loss",
        "train_proto_diversity_loss",
        "train_optim_total_loss",
        "train_optim_dst_loss",
        "train_optim_proto_reg_loss",
        "val_total_loss",
        "val_dst_loss",
        "val_proto_reg_loss",
        "val_proto_reg_loss_raw",
        "val_proto_attract_loss",
        "val_proto_separation_loss",
        "val_proto_diversity_loss",
    ]
    available = [col for col in columns if col in history_df.columns]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col in available:
        ax.plot(history_df["epoch"], history_df[col], marker="o", label=col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_all_loss_curve(history_df, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] skipped: {exc}")
        return
    folds = sorted(history_df["fold"].unique().tolist())
    ncols = 2 if len(folds) > 1 else 1
    nrows = math.ceil(len(folds) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, fold in zip(axes_flat, folds):
        fold_df = history_df[history_df["fold"] == fold].sort_values("epoch")
        ax.plot(fold_df["epoch"], fold_df["train_loss"], color="#1f77b4", linewidth=2.5, label="train loss")
        ax.plot(fold_df["epoch"], fold_df["val_loss"], color="#d62728", linewidth=2.5, label="val loss")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.legend(loc="upper right")
    for ax in axes_flat[len(folds):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def save_all_component_curve(history_df, output_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] skipped: {exc}")
        return
    columns = [
        "train_total_loss",
        "train_dst_loss",
        "train_proto_reg_loss",
        "train_proto_reg_loss_raw",
        "train_proto_attract_loss",
        "train_proto_separation_loss",
        "train_proto_diversity_loss",
        "train_optim_total_loss",
        "train_optim_dst_loss",
        "train_optim_proto_reg_loss",
        "val_total_loss",
        "val_dst_loss",
        "val_proto_reg_loss",
        "val_proto_reg_loss_raw",
        "val_proto_attract_loss",
        "val_proto_separation_loss",
        "val_proto_diversity_loss",
    ]
    available = [col for col in columns if col in history_df.columns]
    if not available:
        return
    folds = sorted(history_df["fold"].unique().tolist())
    ncols = 2 if len(folds) > 1 else 1
    nrows = math.ceil(len(folds) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 5.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, fold in zip(axes_flat, folds):
        fold_df = history_df[history_df["fold"] == fold].sort_values("epoch")
        for col in available:
            ax.plot(fold_df["epoch"], fold_df[col], marker="o", label=col)
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
    for ax in axes_flat[len(folds):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def initialize_prototypes(model, train_df, embeddings, args, fold_id):
    rows = train_df["embedding_row"].astype(int).to_numpy()
    train_embeddings = np.asarray(embeddings[rows], dtype=np.float32)
    labels = train_df[args.label].astype(int).to_numpy()
    warnings_list = model.initialize_from_embeddings(
        train_embeddings,
        labels,
        random_state=int(args.seed) + int(fold_id),
    )
    for warning_text in warnings_list:
        print(f"[Proto Init] WARNING: {warning_text}")
    print(
        f"[Proto Init] fold={fold_id} initialized DST prototypes from "
        f"{len(train_embeddings)} embeddings."
    )


def train_epoch(model, loader, criterion, regularizer, optimizer, scheduler, scaler,
                epoch, total_epochs, args, logger, device):
    model.train()
    total_losses = AverageMeter()
    dst_losses = AverageMeter()
    proto_losses = AverageMeter()
    proto_raw_losses = AverageMeter()
    attract_losses = AverageMeter()
    separation_losses = AverageMeter()
    diversity_losses = AverageMeter()
    start = time.time()
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")

    for step, data in enumerate(tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs} train]")):
        features = data["x"].to(device, non_blocking=True)
        labels = data["y"].to(device, non_blocking=True).long()
        bs = features.size(0)
        optimizer.zero_grad(set_to_none=True)

        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_features = features[mb_start:mb_end]
            mb_labels = labels[mb_start:mb_end]
            mb_size = mb_features.size(0)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(mb_features)
                dst_loss = criterion(outputs, mb_labels)
                proto_terms = regularizer(outputs, model.proto_head, mb_labels)
                proto_loss_raw = proto_terms.get("raw_total_loss", proto_terms["total_loss"])
                proto_loss = args.dst_proto_loss_weight * proto_loss_raw
                total_loss = dst_loss + proto_loss
                scaled_loss = total_loss * (mb_size / bs)

            total_losses.update(total_loss.item(), mb_size)
            dst_losses.update(dst_loss.item(), mb_size)
            proto_losses.update(proto_loss.item(), mb_size)
            proto_raw_losses.update(proto_loss_raw.item(), mb_size)
            attract_losses.update(proto_terms["attract_loss"].item(), mb_size)
            separation_losses.update(proto_terms["separation_loss"].item(), mb_size)
            diversity_losses.update(proto_terms["diversity_loss"].item(), mb_size)
            scaler.scale(scaled_loss).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.print_freq == 0 or step == len(loader) - 1:
            print(
                f"Epoch [{epoch+1}][{step}/{len(loader)}] "
                f"Loss: {total_losses.val:.4f}({total_losses.avg:.4f}) "
                f"dst: {dst_losses.val:.4f}({dst_losses.avg:.4f}) "
                f"proto: {proto_losses.val:.4f}({proto_losses.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                f"Remain: {time_since(start, float(step+1)/len(loader))}"
            )

        if step % args.log_freq == 0 or step == len(loader) - 1:
            idx = step + len(loader) * epoch
            logger.add_scalar("train/iter_total_loss", total_losses.avg, idx)
            logger.add_scalar("train/iter_dst_loss", dst_losses.avg, idx)
            logger.add_scalar("train/iter_proto_reg_loss", proto_losses.avg, idx)
            logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], idx)

    return {
        "total_loss": total_losses.avg,
        "dst_loss": dst_losses.avg,
        "proto_reg_loss": proto_losses.avg,
        "proto_reg_loss_raw": proto_raw_losses.avg,
        "proto_attract_loss": attract_losses.avg,
        "proto_separation_loss": separation_losses.avg,
        "proto_diversity_loss": diversity_losses.avg,
    }


@torch.no_grad()
def valid_epoch(model, loader, criterion, regularizer, args, device):
    model.eval()
    total_losses = AverageMeter()
    dst_losses = AverageMeter()
    proto_losses = AverageMeter()
    proto_raw_losses = AverageMeter()
    attract_losses = AverageMeter()
    separation_losses = AverageMeter()
    diversity_losses = AverageMeter()
    all_probs = []
    all_mass = []
    all_uncertainty = []
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")

    for data in tqdm(loader, desc="[Valid]"):
        features = data["x"].to(device, non_blocking=True)
        labels = data["y"].to(device, non_blocking=True).long()
        bs = features.size(0)
        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_features = features[mb_start:mb_end]
            mb_labels = labels[mb_start:mb_end]
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(mb_features)
                dst_loss = criterion(outputs, mb_labels)
                proto_terms = regularizer(outputs, model.proto_head, mb_labels)
                proto_loss_raw = proto_terms.get("raw_total_loss", proto_terms["total_loss"])
                proto_loss = args.dst_proto_loss_weight * proto_loss_raw
                total_loss = dst_loss + proto_loss
            mb_size = mb_features.size(0)
            total_losses.update(total_loss.item(), mb_size)
            dst_losses.update(dst_loss.item(), mb_size)
            proto_losses.update(proto_loss.item(), mb_size)
            proto_raw_losses.update(proto_loss_raw.item(), mb_size)
            attract_losses.update(proto_terms["attract_loss"].item(), mb_size)
            separation_losses.update(proto_terms["separation_loss"].item(), mb_size)
            diversity_losses.update(proto_terms["diversity_loss"].item(), mb_size)
            all_probs.append(outputs["prob"].detach().cpu().numpy())
            all_mass.append(outputs["dst_mass"].detach().cpu().numpy())
            all_uncertainty.append(outputs["uncertainty"].detach().cpu().numpy())

    if not all_probs:
        raise ValueError("Validation loader produced no batches.")
    return (
        {
            "total_loss": total_losses.avg,
            "dst_loss": dst_losses.avg,
            "proto_reg_loss": proto_losses.avg,
            "proto_reg_loss_raw": proto_raw_losses.avg,
            "proto_attract_loss": attract_losses.avg,
            "proto_separation_loss": separation_losses.avg,
            "proto_diversity_loss": diversity_losses.avg,
        },
        np.concatenate(all_probs),
        np.concatenate(all_mass),
        np.concatenate(all_uncertainty),
    )


def extract_dst_topk(outputs, topk):
    actual_topk = min(int(topk), int(outputs["prototype_evidence"].size(-1)))
    if actual_topk <= 0:
        return {"actual_topk": 0}
    top_evidence, top_idx = torch.topk(outputs["prototype_evidence"], k=actual_topk, dim=-1)
    top_similarity = torch.gather(outputs["prototype_similarity"], dim=-1, index=top_idx)
    return {
        "proto_topk_idx": top_idx,
        "proto_topk_evidence": top_evidence,
        "proto_topk_similarity": top_similarity,
        "actual_topk": actual_topk,
    }


@torch.no_grad()
def predict_all(model_paths, df_all, embeddings, args, device, threshold=0.5):
    dataset = EmbeddingFeatureDataset(df_all, embeddings, args.label)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=embedding_collate_fn,
    )
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    num_classes = int(args.num_classes)
    topk = int(args.dst_proto_topk)

    all_scores = []
    all_probs = []
    all_mass = []
    all_uncertainty = []
    per_fold_results = []

    for model_idx, model_item in enumerate(model_paths):
        if isinstance(model_item, (tuple, list)):
            fold_idx, ckpt_path = model_item
        else:
            fold_idx, ckpt_path = model_idx, model_item
        print(f"[Predict] Loading fold {fold_idx} model: {ckpt_path}")
        model = build_model(args).to(device)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.eval()

        fold_probs = []
        fold_mass = []
        fold_uncertainty = []
        fold_topk_idx = []
        fold_topk_evidence = []
        fold_topk_similarity = []
        actual_topk = 0

        for data in tqdm(loader, desc=f"[Predict] fold {fold_idx}"):
            features = data["x"].to(device, non_blocking=True)
            bs = features.size(0)
            for mb_start in range(0, bs, micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, bs)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(features[mb_start:mb_end])
                topk_outputs = extract_dst_topk(outputs, topk)
                actual_topk = int(topk_outputs["actual_topk"])
                fold_probs.append(outputs["prob"].detach().cpu().numpy())
                fold_mass.append(outputs["dst_mass"].detach().cpu().numpy())
                fold_uncertainty.append(outputs["uncertainty"].detach().cpu().numpy())
                if actual_topk > 0:
                    fold_topk_idx.append(topk_outputs["proto_topk_idx"].detach().cpu().numpy())
                    fold_topk_evidence.append(topk_outputs["proto_topk_evidence"].detach().cpu().numpy())
                    fold_topk_similarity.append(topk_outputs["proto_topk_similarity"].detach().cpu().numpy())

        fold_probability = np.concatenate(fold_probs)
        fold_mass_arr = np.concatenate(fold_mass)
        fold_uncertainty_arr = np.concatenate(fold_uncertainty)
        fold_score = fold_probability[:, 1]
        fold_label = (fold_score >= threshold).astype(int)
        fold_result = {
            "fold": int(fold_idx),
            "pred_score": fold_score,
            "pred_label": fold_label,
            "probability": fold_probability,
            "dst_mass": fold_mass_arr,
            "uncertainty": fold_uncertainty_arr,
            "actual_topk": int(actual_topk),
        }
        if actual_topk > 0:
            fold_result.update(
                {
                    "proto_topk_idx": np.concatenate(fold_topk_idx),
                    "proto_topk_evidence": np.concatenate(fold_topk_evidence),
                    "proto_topk_similarity": np.concatenate(fold_topk_similarity),
                    "proto_topk_source_fold": np.full(
                        np.concatenate(fold_topk_idx).shape,
                        int(fold_idx),
                        dtype=np.int64,
                    ),
                }
            )
        all_scores.append(fold_score)
        all_probs.append(fold_probability)
        all_mass.append(fold_mass_arr)
        all_uncertainty.append(fold_uncertainty_arr)
        per_fold_results.append(fold_result)
        torch.cuda.empty_cache()

    pred_score = np.mean(all_scores, axis=0)
    pred_probability = np.mean(all_probs, axis=0)
    pred_mass = np.mean(all_mass, axis=0)
    pred_uncertainty = np.mean(all_uncertainty, axis=0)
    pred_label = (pred_score >= threshold).astype(int)
    ensemble_result = {
        "pred_score": pred_score,
        "pred_label": pred_label,
        "probability": pred_probability,
        "dst_mass": pred_mass,
        "uncertainty": pred_uncertainty,
        "actual_topk": int(per_fold_results[0].get("actual_topk", 0)),
    }

    actual_topk = int(ensemble_result["actual_topk"])
    if actual_topk > 0 and "proto_topk_idx" in per_fold_results[0]:
        num_samples = pred_score.shape[0]
        ensemble_topk_idx = np.zeros((num_samples, num_classes, actual_topk), dtype=np.int64)
        ensemble_topk_evidence = np.zeros((num_samples, num_classes, actual_topk), dtype=np.float32)
        ensemble_topk_similarity = np.zeros((num_samples, num_classes, actual_topk), dtype=np.float32)
        ensemble_topk_source_fold = np.zeros((num_samples, num_classes, actual_topk), dtype=np.int64)
        fold_ids = np.array([item["fold"] for item in per_fold_results], dtype=np.int64)
        fold_class_mass = np.stack([item["dst_mass"][:, :num_classes] for item in per_fold_results], axis=0)
        for class_idx in range(num_classes):
            source_pos = np.argmax(fold_class_mass[:, :, class_idx], axis=0)
            for sample_idx, fold_pos in enumerate(source_pos.tolist()):
                ensemble_topk_idx[sample_idx, class_idx, :] = per_fold_results[fold_pos]["proto_topk_idx"][sample_idx, class_idx, :]
                ensemble_topk_evidence[sample_idx, class_idx, :] = per_fold_results[fold_pos]["proto_topk_evidence"][sample_idx, class_idx, :]
                ensemble_topk_similarity[sample_idx, class_idx, :] = per_fold_results[fold_pos]["proto_topk_similarity"][sample_idx, class_idx, :]
                ensemble_topk_source_fold[sample_idx, class_idx, :] = fold_ids[fold_pos]
        ensemble_result.update(
            {
                "proto_topk_idx": ensemble_topk_idx,
                "proto_topk_evidence": ensemble_topk_evidence,
                "proto_topk_similarity": ensemble_topk_similarity,
                "proto_topk_source_fold": ensemble_topk_source_fold,
            }
        )

    return ensemble_result, per_fold_results


def save_prediction_csv(df_all, results, output_path, fold_idx=None):
    df_out = df_all.copy()
    df_out["pred_score"] = results["pred_score"]
    df_out["pred_label"] = results["pred_label"]
    df_out["prediction_score"] = results["pred_score"]
    df_out["predicted_class"] = results["pred_label"]
    num_classes = results["probability"].shape[1]
    for class_idx in range(num_classes):
        df_out[f"dst_mass_{class_idx}"] = results["dst_mass"][:, class_idx]
        df_out[f"probability_{class_idx}"] = results["probability"][:, class_idx]
    df_out["dst_mass_omega"] = results["dst_mass"][:, num_classes]
    df_out["uncertainty"] = results["uncertainty"].squeeze()

    actual_topk = int(results.get("actual_topk", 0))
    if actual_topk > 0 and "proto_topk_idx" in results:
        for class_idx in range(num_classes):
            for rank in range(actual_topk):
                suffix = f"proto_c{class_idx}_top{rank+1}"
                df_out[f"{suffix}_idx"] = results["proto_topk_idx"][:, class_idx, rank]
                df_out[f"{suffix}_evidence"] = results["proto_topk_evidence"][:, class_idx, rank]
                df_out[f"{suffix}_similarity"] = results["proto_topk_similarity"][:, class_idx, rank]
                if "proto_topk_source_fold" in results:
                    df_out[f"{suffix}_source_fold"] = results["proto_topk_source_fold"][:, class_idx, rank]

    if "fold" not in df_out.columns:
        df_out["fold"] = -1
    df_out["source_model"] = f"fold{fold_idx}" if fold_idx is not None else "ensemble"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[Save] Predictions saved -> {output_path}")
    return df_out


def build_model(args):
    return DSTPrototypeModel(
        in_features=args.feature_dim,
        num_classes=args.num_classes,
        prototypes_per_class=args.dst_proto_k,
        topk=args.dst_proto_topk,
        normalize=args.dst_proto_normalize,
        gamma_init=args.dst_proto_gamma_init,
        alpha_init=args.dst_proto_alpha_init,
        dropout=args.dst_proto_dropout,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DST-Prototype on pre-extracted Mammo-FM embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embedding-dir", default="./output/origin_embeddings_finetuned_fold0", type=str)
    parser.add_argument("--embeddings-file", default="embeddings.npy", type=str)
    parser.add_argument("--metadata-file", default="metadata.csv", type=str)
    parser.add_argument("--model-save-dir", default=None, type=str)
    parser.add_argument("--csv-output-dir", default=None, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--fold-csv", default=None, type=str)
    parser.add_argument("--use-existing-fold-csv", default="n", type=str)
    parser.add_argument("--overlap-policy", default="test", choices=["error", "test", "train", "training"])
    parser.add_argument("--split-by-cohort", default="y", type=str)
    parser.add_argument("--cohort-col", default="cohort_num", type=str)
    parser.add_argument("--train-cohorts", default="1-8", type=str)
    parser.add_argument("--test-cohorts", default="9-10", type=str)
    parser.add_argument("--label", default="cancer", type=str)
    parser.add_argument("--n_folds", default=0, type=int)
    parser.add_argument("--kfold0-val-frac", default=0.2, type=float)
    parser.add_argument("--kfold0-val-max-frac", default=0.5, type=float)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--early-stop", default=3, type=int)
    parser.add_argument("--early-stop-metric", default="auroc", choices=["auroc", "val_loss"])
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--micro-batch-size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--weighted-BCE", "--weighted_BCE", dest="weighted_BCE", default="y", type=str)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--log-freq", default=200, type=int)
    parser.add_argument("--dst-proto-k", default=10, type=int)
    parser.add_argument("--dst-proto-topk", default=3, type=int)
    parser.add_argument("--dst-proto-normalize", default="y", type=str)
    parser.add_argument("--dst-proto-gamma-init", default=1.0, type=float)
    parser.add_argument("--dst-proto-alpha-init", default=0.0, type=float)
    parser.add_argument("--dst-proto-dropout", default=0.0, type=float)
    parser.add_argument("--dst-proto-attract-weight", default=0.1, type=float)
    parser.add_argument("--dst-proto-separation-weight", default=0.1, type=float)
    parser.add_argument("--dst-proto-diversity-weight", default=0.01, type=float)
    parser.add_argument("--dst-proto-loss-weight", default=1.0, type=float)
    parser.add_argument("--dst-proto-margin", default=1.0, type=float)
    parser.add_argument("--dst-proto-balance-classes", default="y", type=str)
    parser.add_argument("--max-samples", default=None, type=int)
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    args.num_classes = 2
    args.n_folds = int(args.n_folds)
    if args.n_folds < 0 or args.n_folds == 1:
        raise ValueError(f"n_folds must be 0 or >= 2, got {args.n_folds}")
    args.kfold0_val_frac = normalize_fraction_arg(args.kfold0_val_frac, "kfold0_val_frac")
    args.kfold0_val_max_frac = normalize_fraction_arg(args.kfold0_val_max_frac, "kfold0_val_max_frac")
    if args.kfold0_val_max_frac < args.kfold0_val_frac:
        raise ValueError("kfold0_val_max_frac must be >= kfold0_val_frac.")
    args.split_by_cohort = parse_bool(args.split_by_cohort, default=True)
    args.use_existing_fold_csv = parse_bool(args.use_existing_fold_csv, default=False)
    args.weighted_bce = parse_bool(args.weighted_BCE, default=True)
    args.dst_proto_normalize = parse_bool(args.dst_proto_normalize, default=True)
    args.dst_proto_balance_classes = parse_bool(args.dst_proto_balance_classes, default=True)
    args.dst_proto_k = int(args.dst_proto_k)
    args.dst_proto_topk = min(int(args.dst_proto_topk), args.dst_proto_k)
    args.dst_proto_loss_weight = float(args.dst_proto_loss_weight)
    args.apex = parse_bool(args.apex, default=True)
    args.early_stop_metric = str(getattr(args, "early_stop_metric", "auroc")).strip().lower()
    if args.early_stop_metric not in {"auroc", "val_loss"}:
        raise ValueError("early_stop_metric must be 'auroc' or 'val_loss'.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    embedding_dir = resolve_project_path(args.embedding_dir)
    embeddings, metadata, manifest, embeddings_path, metadata_path = load_embedding_cache(
        embedding_dir,
        args.embeddings_file,
        args.metadata_file,
    )
    if args.max_samples is not None:
        metadata = metadata.head(int(args.max_samples)).copy()
    args.feature_dim = int(embeddings.shape[1])

    output_arg = getattr(args, "output_dir", None)
    if args.model_save_dir in (None, "") and output_arg not in (None, ""):
        model_save_dir = resolve_project_path(Path(output_arg) / "checkpoints")
    else:
        model_save_dir = resolve_project_path(args.model_save_dir, "./best_model/dst_proto_features")
    if args.csv_output_dir in (None, "") and output_arg not in (None, ""):
        csv_output_dir = resolve_project_path(output_arg)
    else:
        csv_output_dir = resolve_project_path(args.csv_output_dir, "./output/dst_proto_features")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = model_save_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.model_save_dir = model_save_dir
    args.csv_output_dir = csv_output_dir
    args.output_dir = csv_output_dir

    working_metadata_path = csv_output_dir / "embedding_metadata_for_folds.csv"
    metadata.to_csv(working_metadata_path, index=False)
    if args.fold_csv not in (None, ""):
        fold_csv_path = Path(args.fold_csv)
        if not fold_csv_path.is_absolute():
            fold_csv_path = csv_output_dir / fold_csv_path
    else:
        fold_csv_path = csv_output_dir / f"{metadata_path.stem}_dst_folds.csv"

    if args.use_existing_fold_csv:
        if args.fold_csv in (None, ""):
            raise ValueError("use_existing_fold_csv=y requires --fold-csv.")
        if not fold_csv_path.exists():
            raise FileNotFoundError(f"Requested existing fold CSV does not exist: {fold_csv_path}")
        folds_csv = fold_csv_path
    else:
        folds_csv = create_folds(
            working_metadata_path,
            label_col=args.label,
            n_folds=args.n_folds,
            seed=args.seed,
            output_path=fold_csv_path,
            overlap_policy=args.overlap_policy,
            split_by_cohort=args.split_by_cohort,
            cohort_col=args.cohort_col,
            train_cohorts=args.train_cohorts,
            test_cohorts=args.test_cohorts,
            kfold0_val_frac=args.kfold0_val_frac,
            kfold0_val_max_frac=args.kfold0_val_max_frac,
        )
    df = read_mammo_csv(folds_csv)

    print("=" * 60)
    print("  DST-Prototype Feature Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Model Save Dir: {model_save_dir}")
    print(f"CSV Output Dir: {csv_output_dir}")
    print(f"Folds: {args.n_folds}  Epochs: {args.epochs}  Batch: {args.batch_size}")
    print(f"DST Proto K: {args.dst_proto_k}  TopK: {args.dst_proto_topk}")
    print(f"DST gamma init: {args.dst_proto_gamma_init}  alpha init: {args.dst_proto_alpha_init}")
    print(f"Manifest source: {manifest.get('source', '(unknown)')}")
    print("=" * 60)

    regularizer = PrototypeRegularizationLoss(
        attract_weight=args.dst_proto_attract_weight,
        separation_weight=args.dst_proto_separation_weight,
        diversity_weight=args.dst_proto_diversity_weight,
        margin=args.dst_proto_margin,
        balance_classes=args.dst_proto_balance_classes,
    )

    fold_model_paths = []
    all_histories = []
    data_stem = metadata_path.stem
    fold_ids = [0] if args.n_folds == 0 else list(range(args.n_folds))

    for fold in fold_ids:
        print(f"\n{'=' * 60}\n  Fold {fold}\n{'=' * 60}")
        seed_all(args.seed)
        train_pool_mask = df["split"] == "train"
        if args.n_folds == 0:
            train_df = df[train_pool_mask].reset_index(drop=True)
            valid_df = df[df["split"] == "val"].reset_index(drop=True)
        else:
            train_df = df[train_pool_mask & (df["fold"] != fold)].reset_index(drop=True)
            valid_df = df[train_pool_mask & (df["fold"] == fold)].reset_index(drop=True)
        save_fold_split_csv(df, csv_output_dir / f"{data_stem}_fold{fold}_splits_dst_proto.csv", fold, args.n_folds)

        train_ds = EmbeddingFeatureDataset(train_df, embeddings, args.label)
        valid_ds = EmbeddingFeatureDataset(valid_df, embeddings, args.label)
        if len(train_ds) == 0 or len(valid_ds) == 0:
            print(f"  WARNING: Fold {fold} has empty train or valid data; skipping.")
            continue
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=embedding_collate_fn,
        )
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=embedding_collate_fn,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=embedding_collate_fn,
        )

        model = build_model(args).to(device)
        initialize_prototypes(model, train_df, embeddings, args, fold)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = len(train_loader) if args.warmup_epochs >= 1 else int(args.warmup_epochs * len(train_loader))
        scheduler = make_scheduler(optimizer, total_steps=len(train_loader) * args.epochs, warmup_steps=warmup_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
        class_weights = compute_class_weights(train_df, args.label, args.weighted_bce)
        criterion = PrototypeDSTNLLLoss(weight=class_weights)
        logger = make_summary_writer(log_dir / f"fold{fold}")
        early_stopper = EarlyStopping(patience=args.early_stop) if args.early_stop > 0 else None
        best_score = -float("inf")
        best_auroc = -float("inf")
        best_model_path = model_save_dir / f"best_fold{fold}_seed{args.seed}_dst_proto.pth"
        saved_best = False
        last_epoch = -1
        last_metrics = None
        history_rows = []

        for epoch in range(args.epochs):
            last_epoch = epoch
            start = time.time()
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                regularizer,
                optimizer,
                scheduler,
                scaler,
                epoch,
                args.epochs,
                args,
                logger,
                device,
            )
            train_eval_metrics, _, _, _ = valid_epoch(
                model,
                train_eval_loader,
                criterion,
                regularizer,
                args,
                device,
            )
            val_metrics, probabilities, masses, uncertainties = valid_epoch(
                model,
                valid_loader,
                criterion,
                regularizer,
                args,
                device,
            )
            val_loss = val_metrics["total_loss"]
            pred_score = probabilities[:, 1]
            valid_df_fold = valid_df.copy()
            valid_df_fold["prediction"] = pred_score
            valid_agg = valid_df_fold.groupby("patient_id").agg({args.label: "max", "prediction": "max"})
            metrics = all_classification_metrics(valid_agg[args.label].values, valid_agg["prediction"].values)
            last_metrics = metrics
            monitor_value = float(val_loss) if args.early_stop_metric == "val_loss" else float(metrics["AUROC"])
            monitor_score = -monitor_value if args.early_stop_metric == "val_loss" else monitor_value
            is_best = monitor_score > best_score
            elapsed = time.time() - start
            history_rows.append(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "train_loss": float(train_eval_metrics["total_loss"]),
                    "train_total_loss": float(train_eval_metrics["total_loss"]),
                    "train_dst_loss": float(train_eval_metrics["dst_loss"]),
                    "train_proto_reg_loss": float(train_eval_metrics["proto_reg_loss"]),
                    "train_proto_reg_loss_raw": float(train_eval_metrics["proto_reg_loss_raw"]),
                    "train_proto_attract_loss": float(train_eval_metrics["proto_attract_loss"]),
                    "train_proto_separation_loss": float(train_eval_metrics["proto_separation_loss"]),
                    "train_proto_diversity_loss": float(train_eval_metrics["proto_diversity_loss"]),
                    "train_eval_loss": float(train_eval_metrics["total_loss"]),
                    "train_eval_total_loss": float(train_eval_metrics["total_loss"]),
                    "train_eval_dst_loss": float(train_eval_metrics["dst_loss"]),
                    "train_eval_proto_reg_loss": float(train_eval_metrics["proto_reg_loss"]),
                    "train_eval_proto_reg_loss_raw": float(train_eval_metrics["proto_reg_loss_raw"]),
                    "train_eval_proto_attract_loss": float(train_eval_metrics["proto_attract_loss"]),
                    "train_eval_proto_separation_loss": float(train_eval_metrics["proto_separation_loss"]),
                    "train_eval_proto_diversity_loss": float(train_eval_metrics["proto_diversity_loss"]),
                    "train_optim_loss": float(train_metrics["total_loss"]),
                    "train_optim_total_loss": float(train_metrics["total_loss"]),
                    "train_optim_dst_loss": float(train_metrics["dst_loss"]),
                    "train_optim_proto_reg_loss": float(train_metrics["proto_reg_loss"]),
                    "train_optim_proto_reg_loss_raw": float(train_metrics["proto_reg_loss_raw"]),
                    "train_optim_proto_attract_loss": float(train_metrics["proto_attract_loss"]),
                    "train_optim_proto_separation_loss": float(train_metrics["proto_separation_loss"]),
                    "train_optim_proto_diversity_loss": float(train_metrics["proto_diversity_loss"]),
                    "val_loss": float(val_loss),
                    "val_total_loss": float(val_metrics["total_loss"]),
                    "val_dst_loss": float(val_metrics["dst_loss"]),
                    "val_proto_reg_loss": float(val_metrics["proto_reg_loss"]),
                    "val_proto_reg_loss_raw": float(val_metrics["proto_reg_loss_raw"]),
                    "val_proto_attract_loss": float(val_metrics["proto_attract_loss"]),
                    "val_proto_separation_loss": float(val_metrics["proto_separation_loss"]),
                    "val_proto_diversity_loss": float(val_metrics["proto_diversity_loss"]),
                    "auroc": float(metrics["AUROC"]),
                    "auprc": float(metrics["AUPRC"]),
                    "bacc": float(metrics["bACC"]),
                    "early_stop_metric": args.early_stop_metric,
                    "monitor_value": float(monitor_value),
                    "monitor_score": float(monitor_score),
                    "mean_uncertainty": float(np.mean(uncertainties)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_sec": float(elapsed),
                    "is_best": int(is_best),
                }
            )
            print(
                f"Fold {fold} Epoch {epoch+1} - train_eval_loss: {train_eval_metrics['total_loss']:.4f} "
                f"train_optim_loss: {train_metrics['total_loss']:.4f} "
                f"train_eval_dst: {train_eval_metrics['dst_loss']:.4f} "
                f"train_eval_proto: {train_eval_metrics['proto_reg_loss']:.4f} "
                f"val_loss: {val_loss:.4f} val_dst: {val_metrics['dst_loss']:.4f} "
                f"val_proto: {val_metrics['proto_reg_loss']:.4f} AUROC: {metrics['AUROC']:.4f} "
                f"AUPRC: {metrics['AUPRC']:.4f} bACC: {metrics['bACC']:.4f} time: {elapsed:.0f}s"
            )
            logger.add_scalar("train/epoch_total_loss", train_metrics["total_loss"], epoch + 1)
            logger.add_scalar("train/epoch_dst_loss", train_metrics["dst_loss"], epoch + 1)
            logger.add_scalar("train_eval/epoch_total_loss", train_eval_metrics["total_loss"], epoch + 1)
            logger.add_scalar("train_eval/epoch_dst_loss", train_eval_metrics["dst_loss"], epoch + 1)
            logger.add_scalar("train_eval/epoch_proto_reg_loss", train_eval_metrics["proto_reg_loss"], epoch + 1)
            logger.add_scalar("train_eval/epoch_proto_reg_loss_raw", train_eval_metrics["proto_reg_loss_raw"], epoch + 1)
            logger.add_scalar("train_eval/epoch_proto_attract_loss", train_eval_metrics["proto_attract_loss"], epoch + 1)
            logger.add_scalar("train_eval/epoch_proto_separation_loss", train_eval_metrics["proto_separation_loss"], epoch + 1)
            logger.add_scalar("train_eval/epoch_proto_diversity_loss", train_eval_metrics["proto_diversity_loss"], epoch + 1)
            logger.add_scalar("valid/epoch_total_loss", val_metrics["total_loss"], epoch + 1)
            logger.add_scalar("valid/epoch_dst_loss", val_metrics["dst_loss"], epoch + 1)
            logger.add_scalar("valid/epoch_proto_reg_loss", val_metrics["proto_reg_loss"], epoch + 1)
            logger.add_scalar("valid/epoch_proto_reg_loss_raw", val_metrics["proto_reg_loss_raw"], epoch + 1)
            logger.add_scalar("valid/epoch_proto_attract_loss", val_metrics["proto_attract_loss"], epoch + 1)
            logger.add_scalar("valid/epoch_proto_separation_loss", val_metrics["proto_separation_loss"], epoch + 1)
            logger.add_scalar("valid/epoch_proto_diversity_loss", val_metrics["proto_diversity_loss"], epoch + 1)
            logger.add_scalar("valid/AUROC", metrics["AUROC"], epoch + 1)
            logger.add_scalar("valid/AUPRC", metrics["AUPRC"], epoch + 1)
            logger.add_scalar("valid/bACC", metrics["bACC"], epoch + 1)
            logger.add_scalar("valid/loss", val_loss, epoch + 1)

            if is_best:
                best_score = monitor_score
                best_auroc = metrics["AUROC"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "auroc": best_auroc,
                        "best_score": best_score,
                        "best_metric": args.early_stop_metric,
                        "best_metric_value": monitor_value,
                        "metrics": metrics,
                        "dst_config": {
                            "feature_dim": args.feature_dim,
                            "dst_proto_k": args.dst_proto_k,
                            "dst_proto_topk": args.dst_proto_topk,
                            "dst_proto_normalize": args.dst_proto_normalize,
                            "dst_proto_gamma_init": args.dst_proto_gamma_init,
                            "dst_proto_alpha_init": args.dst_proto_alpha_init,
                            "dst_proto_dropout": args.dst_proto_dropout,
                            "dst_proto_loss_weight": args.dst_proto_loss_weight,
                            "embedding_dir": str(embedding_dir),
                        },
                    },
                    best_model_path,
                )
                saved_best = True
                print(
                    f"  -> Saved best ({args.early_stop_metric}={monitor_value:.4f}, "
                    f"AUROC={best_auroc:.4f})"
                )

            if early_stopper is not None and early_stopper(monitor_score):
                best_value = -early_stopper.best_score if args.early_stop_metric == "val_loss" else early_stopper.best_score
                print(
                    f"  -> Early stopping at epoch {epoch+1} "
                    f"(best {args.early_stop_metric}={best_value:.4f})"
                )
                break

        if not saved_best:
            torch.save(
                {
                    "epoch": last_epoch,
                    "model": model.state_dict(),
                    "auroc": best_auroc,
                    "best_score": best_score,
                    "best_metric": args.early_stop_metric,
                    "metrics": last_metrics or {},
                    "dst_config": {"feature_dim": args.feature_dim},
                },
                best_model_path,
            )

        if history_rows:
            history_df = pd.DataFrame(history_rows)
            history_csv = csv_output_dir / f"{data_stem}_fold{fold}_loss_history_dst_proto.csv"
            history_png = csv_output_dir / f"{data_stem}_fold{fold}_loss_curve_dst_proto.png"
            component_png = csv_output_dir / f"{data_stem}_fold{fold}_loss_components_dst_proto.png"
            history_df.to_csv(history_csv, index=False)
            save_loss_curve(history_df, history_png, title=f"DST k={args.dst_proto_k} - fold {fold}")
            save_component_curve(history_df, component_png, title=f"Fold {fold} DST-Prototype Components")
            all_histories.append(history_df)
            print(f"  Saved fold history -> {history_csv}")

        if best_model_path.exists():
            fold_model_paths.append((fold, best_model_path))
        logger.close()
        del model, optimizer, scheduler, scaler, criterion
        torch.cuda.empty_cache()
        gc.collect()

    if all_histories:
        all_history = pd.concat(all_histories, ignore_index=True)
        all_history_csv = csv_output_dir / f"{data_stem}_all_folds_loss_history_dst_proto.csv"
        all_history_png = csv_output_dir / f"{data_stem}_all_folds_loss_curve_dst_proto.png"
        all_component_png = csv_output_dir / f"{data_stem}_all_folds_loss_components_dst_proto.png"
        all_history.to_csv(all_history_csv, index=False)
        save_all_loss_curve(all_history, all_history_png, title=f"DST k={args.dst_proto_k}")
        save_all_component_curve(all_history, all_component_png, title="All Folds DST-Prototype Components")
        print(f"\n[Loss] All folds history saved -> {all_history_csv}")
        print(f"[Loss] All folds curves saved  -> {all_history_png}")
        print(f"[Loss] All folds component curves saved -> {all_component_png}")

    print(f"\n{'=' * 60}")
    print("  Ensemble prediction on ALL data (DST-Prototype)")
    print(f"{'=' * 60}")
    if not fold_model_paths:
        print("[ERROR] No fold models were saved! Cannot run prediction.")
        return
    df_all = read_mammo_csv(folds_csv)
    ensemble_results, per_fold_results = predict_all(
        fold_model_paths,
        df_all,
        embeddings,
        args,
        device,
        threshold=0.5,
    )
    for fold_result in per_fold_results:
        fold_id = fold_result["fold"]
        save_prediction_csv(
            df_all,
            fold_result,
            csv_output_dir / f"{data_stem}_predictions_dst_proto_fold{fold_id}.csv",
            fold_idx=fold_id,
        )
    ensemble_csv = csv_output_dir / f"{data_stem}_predictions_dst_proto_ensemble.csv"
    ensemble_df = save_prediction_csv(df_all, ensemble_results, ensemble_csv, fold_idx=None)
    print(f"\n[Final] Ensemble DST-Prototype predictions saved -> {ensemble_csv}")
    print(
        f"[Final] pred_score stats: mean={ensemble_results['pred_score'].mean():.4f} "
        f"std={ensemble_results['pred_score'].std():.4f} "
        f"min={ensemble_results['pred_score'].min():.4f} max={ensemble_results['pred_score'].max():.4f}"
    )
    print(f"[Final] pred_label distribution:\n{ensemble_df['pred_label'].value_counts()}")
    if args.label in ensemble_df.columns:
        test_df = ensemble_df[ensemble_df["split"] == "test"]
        if len(test_df) > 0:
            metrics = all_classification_metrics(test_df[args.label].values, test_df["pred_score"].values)
            metrics["dst_proto_loss_weight"] = float(args.dst_proto_loss_weight)
            print("\n[Final] Test set metrics:")
            print(f"  AUROC: {metrics['AUROC']:.4f}")
            print(f"  AUPRC: {metrics['AUPRC']:.4f}")
            print(f"  bACC:  {metrics['bACC']:.4f}")
            save_metrics_csv(metrics, csv_output_dir / f"{data_stem}_test_metrics_dst_proto.csv", split_name="test")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
