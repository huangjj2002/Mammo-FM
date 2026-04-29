
import sys

if "--help" in sys.argv or "-h" in sys.argv:
    import argparse
    _p = argparse.ArgumentParser(
        description="5-Fold CV fine-tuning for breast cancer detection using Mammo-FM",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _p.add_argument("--data-dir", default="/mnt/g/data", type=str,
                    help="Root data directory. --img-dir and --csv-file can be relative to this directory (default: /mnt/g/data)")
    _p.add_argument("--img-dir", default="images_png", type=str,
                    help="Image directory. Relative to --data-dir or an absolute path (default: images_png)")
    _p.add_argument("--csv-file", default="train_with_test_data.csv", type=str,
                    help="CSV data file. Relative to --data-dir or an absolute path (default: train_with_test_data.csv)")
    _p.add_argument("--clip_chk_pt_path",
                    default="/mnt/g/Mammo_CLIP_PROJECT/Mammo_FM/Mammo-FM-main/model/Mammo-FM_BatmanlabTrained_CLIP.tar",
                    type=str, help="Path to pre-trained Mammo-FM checkpoint (.tar)")
    _p.add_argument("--output-dir",
                    default="/mnt/g/Mammo_CLIP_PROJECT/Mammo_FM/Mammo-FM-main/output/finetune_cancer",
                    type=str, help="Directory for checkpoints, logs, and predictions")
    _p.add_argument("--fold-csv", default=None, type=str,
                    help="Path to save the CSV with fold column. Default: next to --csv-file as *_folds.csv")
 
    _p.add_argument("--dataset", default="Custom", type=str, help="Dataset name, for logging only (default: Custom)")
    _p.add_argument("--data_frac", default="1.0", type=str, help="Fraction of training data to use (default: 1.0)")
    _p.add_argument("--label", default="cancer", type=str, help="Label column name in CSV (default: cancer)")
    _p.add_argument("--arch", default="breast_clip_det_b5_period_n_ft",
                    choices=["breast_clip_det_b5_period_n_lp", "breast_clip_det_b5_period_n_ft"],
                    help="lp=linear probe (frozen backbone), ft=full fine-tuning (default: breast_clip_det_b5_period_n_ft)")

    _p.add_argument("--n_folds", default=5, type=int, help="Number of CV folds (default: 5)")
    _p.add_argument("--epochs", default=10, type=int, help="Max epochs per fold (default: 10)")
    _p.add_argument("--early-stop", default=5, type=int, help="Early stopping patience, 0=disabled (default: 5)")
    _p.add_argument("--batch-size", default=4, type=int, help="Batch size (default: 4)")
    _p.add_argument("--micro-batch-size", default=1, type=int,
                    help="GPU micro-batch size for memory-safe forward/backward (default: 1)")
    _p.add_argument("--lr", default=5e-5, type=float, help="Learning rate (default: 5e-5)")
    _p.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay (default: 1e-4)")
    _p.add_argument("--warmup-epochs", default=1, type=float, help="Warmup epochs (default: 1)")
    _p.add_argument("--weighted-BCE", default="n", choices=["y", "n"],
                    help="Use weighted BCE loss for class imbalance (default: n)")
    _p.add_argument("--img-size", nargs="+", default=[1520, 912], type=int, help="Image size [H W] (default: 1520 912)")
    _p.add_argument("--seed", default=42, type=int, help="Random seed (default: 42)")
    _p.add_argument("--num-workers", default=2, type=int, help="DataLoader num_workers (default: 2)")

    _p.add_argument("--gpu-id", default=0, type=int, help="GPU device ID to use (default: 0)")
    _p.add_argument("--device", default="cuda", type=str, help="Device: cuda or cpu (default: cuda)")
    _p.add_argument("--apex", default="y", type=str, help="AMP mixed precision: y/n (default: y)")
    _p.add_argument("--print-freq", default=50, type=int, help="Print training stats every N steps (default: 50)")
    _p.add_argument("--log-freq", default=200, type=int, help="Log to TensorBoard every N steps (default: 200)")

    _p.add_argument("--alpha", default=10, type=float, help="ElasticTransform alpha (default: 10)")
    _p.add_argument("--sigma", default=15, type=float, help="ElasticTransform sigma (default: 15)")
    _p.add_argument("--p", default=1.0, type=float, help="Augmentation probability (default: 1.0)")

    _p.add_argument("--mean", default=0.3089279, type=float, help="Image normalization mean (default: 0.3089279)")
    _p.add_argument("--std", default=0.25053555408335154, type=float, help="Image normalization std (default: 0.25053555408335154)")
    _p.parse_args(["--help"])



import argparse
import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from albumentations import Compose, HorizontalFlip, VerticalFlip, Affine, ElasticTransform, Resize
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "src" / "codebase"))

from Classifiers.models.breast_clip_classifier import BreastClipClassifier
from metrics import all_classification_metrics, compute_opt_thres
from utils import seed_all, AverageMeter, timeSince
from breastclip.scheduler import LinearWarmupCosineAnnealingLR




class CustomMammoDataset(Dataset):
    """Dataset for breast cancer classification from custom CSV + PNG images."""
    def __init__(self, df, img_root, label_col="cancer", transform=None, mean=0.3089279, std=0.25053555408335154):
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
                return True  # stop
            return False

    def reset(self):
        self.best_score = -float("inf")
        self.counter = 0




def create_folds(csv_path, label_col="cancer", n_folds=5, seed=42, output_path=None):

    rng = np.random.RandomState(seed)

    df = pd.read_csv(csv_path)
    required_cols = {"split", "patient_id", "image_id", label_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing_cols)}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    print(f"[Folds] Loaded {len(df)} rows. split counts:\n{df['split'].value_counts()}")

    df["fold"] = -1

    train_mask = df["split"] == "training"
    train_df = df[train_mask].copy()
    if train_df.empty:
        raise ValueError("No rows with split == 'training'; cannot create CV folds.")


    patient_info = train_df.groupby("patient_id").agg(
        label=(label_col, "max"),
        n_images=("image_id", "count"),
    ).reset_index()
    patient_info["label"] = patient_info["label"].astype(int)

    pos_patients = patient_info[patient_info["label"] == 1]["patient_id"].values
    neg_patients = patient_info[patient_info["label"] == 0]["patient_id"].values
    if len(patient_info) < n_folds:
        raise ValueError(
            f"n_folds={n_folds} is larger than the number of training patients "
            f"({len(patient_info)}). Reduce n_folds."
        )
    print(f"[Folds] Training patients: {len(patient_info)} "
          f"(pos={len(pos_patients)}, neg={len(neg_patients)})")
    print(f"[Folds] Training images:   {len(train_df)} "
          f"(pos={(train_df[label_col]==1).sum()}, neg={(train_df[label_col]==0).sum()})")

    if len(pos_patients) < n_folds:
        print(f"[Folds] WARNING: Only {len(pos_patients)} positive patient(s) for {n_folds} folds. "
              f"Some folds would have 0 cancer cases. "
              f"Falling back to manual stratified split.")
        use_manual = True
    else:
        use_manual = False


    if use_manual:
        rng.shuffle(pos_patients)
        rng.shuffle(neg_patients)

        pos_folds = np.array_split(pos_patients, n_folds)
        neg_folds = np.array_split(neg_patients, n_folds)

        for fold_id in range(n_folds):
            fold_patients = np.concatenate([pos_folds[fold_id], neg_folds[fold_id]])
            fold_mask = train_df["patient_id"].isin(fold_patients)
            df.loc[train_df[fold_mask].index, "fold"] = fold_id


    else:
        patient_ids = patient_info["patient_id"].values
        y_patient = patient_info["label"].values.astype(int)

        try:
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_id, (_, val_patient_pos) in enumerate(
                sgkf.split(patient_ids, y_patient, groups=patient_ids)
            ):
                val_patients = set(patient_ids[val_patient_pos])
                val_mask = train_df["patient_id"].isin(val_patients)
                df.loc[train_df[val_mask].index, "fold"] = fold_id
        except ValueError as e:
            print(f"[Folds] StratifiedGroupKFold failed ({e}). Falling back to manual split.")
            rng.shuffle(pos_patients)
            rng.shuffle(neg_patients)
            pos_folds = np.array_split(pos_patients, n_folds)
            neg_folds = np.array_split(neg_patients, n_folds)
            for fold_id in range(n_folds):
                fold_patients = np.concatenate([pos_folds[fold_id], neg_folds[fold_id]])
                fold_mask = train_df["patient_id"].isin(fold_patients)
                df.loc[train_df[fold_mask].index, "fold"] = fold_id


    fold_ok = True
    for f in range(n_folds):
        f_pos_patients = df[(df["fold"] == f) & (df[label_col] == 1)]["patient_id"].nunique()
        f_neg_patients = df[(df["fold"] == f) & (df[label_col] == 0)]["patient_id"].nunique()
        if f_pos_patients == 0:
            print(f"[Folds] ERROR: Fold {f} has ZERO positive patients! "
                  f"Consider reducing --n_folds. Got {len(pos_patients)} pos patients total.")
            fold_ok = False
        if f_neg_patients == 0:
            print(f"[Folds] ERROR: Fold {f} has ZERO negative patients! "
                  f"Consider reducing --n_folds. Got {len(neg_patients)} neg patients total.")
            fold_ok = False
    if not fold_ok:
        print("[Folds] FOLD BALANCE CHECK WARNING. Continuing because patient-level grouping is prioritized.")


    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_folds{csv_path.suffix}"
    else:
        output_path = Path(output_path)

    df.to_csv(output_path, index=False)
    print(f"[Folds] Saved {n_folds}-fold (patient-grouped) CSV → {output_path}")
    for f in range(n_folds):
        fold_count = (df["fold"] == f).sum()
        pos_count = ((df["fold"] == f) & (df[label_col] == 1)).sum()
        patient_count = df[df["fold"] == f]["patient_id"].nunique()
        pos_patient_count = df[(df["fold"] == f) & (df[label_col] == 1)]["patient_id"].nunique()
        print(f"  Fold {f}: {fold_count} images, {patient_count} patients "
              f"(pos_patients={pos_patient_count}), cancer_images={pos_count} "
              f"({pos_count / max(fold_count, 1) * 100:.1f}%)")
    print(f"  Test (fold=-1): {(df['fold'] == -1).sum()} images, "
          f"{df[df['fold'] == -1]['patient_id'].nunique()} patients")
    return output_path




def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch, total_epochs, args, logger, device):
    model.train()
    losses = AverageMeter()
    start = time.time()
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")

    for step, data in enumerate(tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs} train]")):
        inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
        labels = data["y"].to(device, non_blocking=True)
        bs = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)
        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_inputs = inputs[mb_start:mb_end]
            mb_labels = labels[mb_start:mb_end]
            mb_size = mb_inputs.size(0)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                preds = model(mb_inputs)
                loss = criterion(preds.view(-1, 1), mb_labels.view(-1, 1))
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
    model.eval()
    losses = AverageMeter()
    all_preds = []
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")

    for data in tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs} valid]"):
        inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
        labels = data["y"].to(device, non_blocking=True)
        bs = inputs.size(0)

        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_inputs = inputs[mb_start:mb_end]
            mb_labels = labels[mb_start:mb_end]
            mb_size = mb_inputs.size(0)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                preds = model(mb_inputs)
                loss = criterion(preds.view(-1, 1), mb_labels.view(-1, 1))
            losses.update(loss.item(), mb_size)
            all_preds.append(preds.sigmoid().squeeze(1).cpu().numpy())

    if not all_preds:
        raise ValueError("Validation loader produced no batches.")
    predictions = np.concatenate(all_preds)
    return losses.avg, predictions




@torch.no_grad()
def predict_all(model_paths, df_all, img_dir, args, device, threshold=None):


    base_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if base_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["model_type"]

    all_scores = []
    per_fold_outputs = []
    th = 0.5 if threshold is None else float(threshold)
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")

    for model_idx, model_item in enumerate(model_paths):
        if isinstance(model_item, (tuple, list)):
            fold_idx, ckpt_path = model_item
        else:
            fold_idx, ckpt_path = model_idx, model_item
        print(f"[Predict] Loading fold {fold_idx} model: {ckpt_path}")

        model = BreastClipClassifier(args, ckpt=base_ckpt, n_class=1)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model = model.to(device)
        model.eval()

        dataset = CustomMammoDataset(df_all, img_dir, transform=None, mean=args.mean, std=args.std)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, collate_fn=collate_fn)

        fold_scores = []
        for data in tqdm(loader, desc=f"[Predict] fold {fold_idx}"):
            inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
            bs = inputs.size(0)
            for mb_start in range(0, bs, micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, bs)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    preds = model(inputs[mb_start:mb_end])
                fold_scores.append(preds.sigmoid().squeeze(1).cpu().numpy())
        fold_score = np.concatenate(fold_scores)
        fold_label = (fold_score >= th).astype(int)
        all_scores.append(fold_score)
        per_fold_outputs.append({
            "fold": fold_idx,
            "pred_score": fold_score,
            "pred_label": fold_label,
        })

        torch.cuda.empty_cache()


    pred_score = np.mean(all_scores, axis=0)


    pred_label = (pred_score >= th).astype(int)
    print(f"[Predict] Binarization threshold: {th:.4f}")

    return pred_score, pred_label, per_fold_outputs




def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(
            description="5-Fold CV fine-tuning for breast cancer detection using Mammo-FM",
            formatter_class=argparse.RawTextHelpFormatter,
        )


        parser.add_argument("--data-dir",
                            default="/mnt/g/data",
                            type=str,
                            help="Root data directory. --img-dir and --csv-file can be "
                                 "relative to this directory (default: /mnt/g/data)")
        parser.add_argument("--img-dir",
                            default="images_png",
                            type=str,
                            help="Image directory. Relative to --data-dir or an absolute path "
                                 "(default: images_png)")
        parser.add_argument("--csv-file",
                            default="train_with_test_data.csv",
                            type=str,
                            help="CSV data file. Relative to --data-dir or an absolute path "
                                 "(default: train_with_test_data.csv)")
        parser.add_argument("--clip_chk_pt_path",
                            default="/mnt/g/Mammo_CLIP_PROJECT/Mammo_FM/Mammo-FM-main/model/Mammo-FM_BatmanlabTrained_CLIP.tar",
                            type=str,
                            help="Path to pre-trained Mammo-FM checkpoint (.tar)")
        parser.add_argument("--output-dir",
                            default="/mnt/g/Mammo_CLIP_PROJECT/Mammo_FM/Mammo-FM-main/output/finetune_cancer",
                            type=str,
                            help="Directory for checkpoints, logs, and predictions")
        parser.add_argument("--fold-csv",
                            default=None,
                            type=str,
                            help="Path to save the CSV with fold column. "
                                 "Default: next to --csv-file as *_folds.csv")


        parser.add_argument("--dataset",
                            default="Custom",
                            type=str,
                            help="Dataset name, for logging only (default: Custom)")
        parser.add_argument("--data_frac",
                            default="1.0",
                            type=str,
                            help="Fraction of training data to use (default: 1.0)")
        parser.add_argument("--label",
                            default="cancer",
                            type=str,
                            help="Label column name in CSV (default: cancer)")
        parser.add_argument("--arch",
                            default="breast_clip_det_b5_period_n_ft",
                            choices=["breast_clip_det_b5_period_n_lp", "breast_clip_det_b5_period_n_ft"],
                            help="lp=linear probe (frozen backbone), ft=full fine-tuning "
                                 "(default: breast_clip_det_b5_period_n_ft)")


        parser.add_argument("--n_folds", default=5, type=int,
                            help="Number of CV folds (default: 5)")
        parser.add_argument("--epochs", default=10, type=int,
                            help="Max epochs per fold (default: 10)")
        parser.add_argument("--early-stop", default=5, type=int,
                            help="Early stopping patience, 0=disabled (default: 5)")
        parser.add_argument("--batch-size", default=4, type=int,
                            help="Batch size (default: 4)")
        parser.add_argument("--micro-batch-size", default=1, type=int,
                            help="GPU micro-batch size for memory-safe forward/backward (default: 1)")
        parser.add_argument("--lr", default=5e-5, type=float,
                            help="Learning rate (default: 5e-5)")
        parser.add_argument("--weight-decay", default=1e-4, type=float,
                            help="Weight decay (default: 1e-4)")
        parser.add_argument("--warmup-epochs", default=1, type=float,
                            help="Warmup epochs (default: 1)")
        parser.add_argument("--weighted-BCE", default="n", choices=["y", "n"],
                            help="Use weighted BCE loss for class imbalance (default: n)")
        parser.add_argument("--img-size", nargs="+", default=[1520, 912], type=int,
                            help="Image size [H W] (default: 1520 912)")
        parser.add_argument("--seed", default=42, type=int,
                            help="Random seed (default: 42)")
        parser.add_argument("--num-workers", default=2, type=int,
                            help="DataLoader num_workers (default: 2)")


        parser.add_argument("--gpu-id", default=0, type=int,
                            help="GPU device ID to use (default: 0)")
        parser.add_argument("--device", default="cuda", type=str,
                            help="Device: cuda or cpu (default: cuda)")
        parser.add_argument("--apex", default="y", type=str,
                            help="AMP mixed precision: y/n (default: y)")
        parser.add_argument("--print-freq", default=50, type=int,
                            help="Print training stats every N steps (default: 50)")
        parser.add_argument("--log-freq", default=200, type=int,
                            help="Log to TensorBoard every N steps (default: 200)")


        parser.add_argument("--alpha", default=10, type=float,
                            help="ElasticTransform alpha (default: 10)")
        parser.add_argument("--sigma", default=15, type=float,
                            help="ElasticTransform sigma (default: 15)")
        parser.add_argument("--p", default=1.0, type=float,
                            help="Augmentation probability (default: 1.0)")


        parser.add_argument("--mean", default=0.3089279, type=float,
                            help="Image normalization mean (default: 0.3089279)")
        parser.add_argument("--std", default=0.25053555408335154, type=float,
                            help="Image normalization std (default: 0.25053555408335154)")

        args = parser.parse_args()


    # ---- 设置可见 GPU（必须在 torch.cuda 相关操作之前） ----
    gpu_id = int(getattr(args, "gpu_id", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    data_dir = Path(args.data_dir)

    if os.path.isabs(args.img_dir):
        args.img_dir = Path(args.img_dir)
    else:
        args.img_dir = data_dir / args.img_dir

    if os.path.isabs(args.csv_file):
        args.data_csv = Path(args.csv_file)
    else:
        args.data_csv = data_dir / args.csv_file


    args.apex = str(args.apex).lower() == "y"
    args.weighted_bce = str(args.weighted_BCE).lower() == "y"
    args.data_frac = float(args.data_frac)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Data CSV: {args.data_csv}")
    print(f"Image dir: {args.img_dir}")
    print(f"Checkpoint: {args.clip_chk_pt_path}")
    print(f"Arch: {args.arch}  |  Folds: {args.n_folds}  |  EarlyStop: {args.early_stop}")
    print(f"Epochs(max): {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Dataset: {args.dataset}  |  Label: {args.label}  |  Data frac: {args.data_frac}")
    print(f"Weighted BCE: {args.weighted_bce}  |  AMP: {args.apex}")
    print("=" * 60)


    folds_csv = create_folds(args.data_csv, label_col=args.label,
                             n_folds=args.n_folds, seed=args.seed,
                             output_path=args.fold_csv)
    df = pd.read_csv(folds_csv)


    base_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if base_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["model_type"]

    fold_model_paths = []   
    oof_parts = []          

    for fold in range(args.n_folds):
        print(f"\n{'=' * 60}")
        print(f"  Fold {fold} / {args.n_folds}")
        print(f"{'=' * 60}")

        seed_all(args.seed)  


        train_df = df[(df["fold"] != -1) & (df["fold"] != fold)].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)

   
        if args.data_frac < 1.0:
            train_df = train_df.sample(frac=args.data_frac, random_state=args.seed).reset_index(drop=True)

        print(f"Train: {len(train_df)}  |  Valid: {len(valid_df)}")
        print(f"  Train cancer%: {train_df[args.label].mean()*100:.1f}%")
        print(f"  Valid cancer%: {valid_df[args.label].mean()*100:.1f}%")

    
        n_valid_classes = valid_df[args.label].nunique()
        if n_valid_classes < 2:
            print(f"  WARNING: Valid set for fold {fold} has only {n_valid_classes} class(es). "
                  f"AUROC will be 0.5 (not meaningful). Consider reducing n_folds or using more data.")

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


        model = BreastClipClassifier(args, ckpt=base_ckpt, n_class=1)
        model = model.to(device)
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = len(train_loader) if args.warmup_epochs >= 1 else int(args.warmup_epochs * len(train_loader))
        lr_config = {
            "total_epochs": args.epochs,
            "warmup_steps": warmup_steps,
            "total_steps": len(train_loader) * args.epochs,
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex)

        if args.weighted_bce:
            pos_count = (train_df[args.label] == 1).sum()
            neg_count = (train_df[args.label] == 0).sum()
            pos_weight = neg_count / max(pos_count, 1)
            print(f"pos_weight: {pos_weight:.2f}")
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        logger = SummaryWriter(log_dir / f"fold{fold}")
        early_stopper = EarlyStopping(patience=args.early_stop) if args.early_stop > 0 else None
        best_auroc = -float("inf")
        best_model_path = ckpt_dir / f"best_fold{fold}_seed{args.seed}.pth"
        saved_best = False  
        best_predictions = None
        best_metrics = None
        last_predictions = None
        last_metrics = None
        last_epoch = -1

        try:
            for epoch in range(args.epochs):
                last_epoch = epoch
                start = time.time()
                train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler,
                                         epoch, args.epochs, args, logger, device)
                val_loss, predictions = valid_epoch(model, valid_loader, criterion, epoch, args.epochs, args, device)
                last_predictions = predictions.copy()

                valid_df["prediction"] = predictions
                valid_agg = valid_df[["patient_id", args.label, "prediction"]].groupby("patient_id").max()
                metrics = all_classification_metrics(valid_agg[args.label].values, valid_agg["prediction"].values)
                last_metrics = metrics
                elapsed = time.time() - start

                print(f"Fold {fold} Epoch {epoch+1} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                      f"AUROC: {metrics['AUROC']:.4f}  time: {elapsed:.0f}s")

                logger.add_scalar("valid/AUROC", metrics["AUROC"], epoch + 1)
                logger.add_scalar("valid/loss", val_loss, epoch + 1)

                if metrics["AUROC"] > best_auroc:
                    best_auroc = metrics["AUROC"]
                    best_predictions = predictions.copy()
                    best_metrics = metrics
                    torch.save({"epoch": epoch, "model": model.state_dict(),
                                "auroc": best_auroc, "predictions": predictions,
                                "metrics": metrics}, best_model_path)
                    saved_best = True
                    print(f"  -> Saved best (AUROC={best_auroc:.4f})")

                if early_stopper is not None and early_stopper(metrics["AUROC"]):
                    print(f"  -> Early stopping at epoch {epoch+1} (best AUROC={early_stopper.best_score:.4f})")
                    break


            if not saved_best:
                print(f"  WARNING: No best model was saved for fold {fold}. "
                      f"Saving last epoch as fallback.")
                best_predictions = last_predictions
                best_metrics = last_metrics
                torch.save({"epoch": last_epoch, "model": model.state_dict(),
                            "auroc": best_auroc, "predictions": best_predictions,
                            "metrics": best_metrics}, best_model_path)

        except Exception as e:
            print(f"  ERROR: Fold {fold} training failed with exception: {e}")
    
            if not saved_best:
                try:
                    torch.save({"epoch": -1, "model": model.state_dict(),
                                "auroc": -1.0, "predictions": None,
                                "metrics": {}}, best_model_path)
                    print(f"  Saved emergency checkpoint for fold {fold}")
                except Exception as e2:
                    print(f"  Could not save emergency checkpoint: {e2}")

        if best_model_path.exists():
            fold_model_paths.append((fold, best_model_path))
            if best_predictions is not None:
                oof_part = valid_df.copy()
                oof_part["oof_pred_score"] = best_predictions
                oof_part["oof_fold"] = fold
                oof_parts.append(oof_part)
        else:
            print(f"  WARNING: No checkpoint for fold {fold}, skipping in ensemble.")
        logger.close()
        del model, optimizer, scheduler, scaler, criterion
        torch.cuda.empty_cache()
        gc.collect()


    print(f"\n{'=' * 60}")
    print("  Ensemble prediction on ALL data")
    print(f"{'=' * 60}")

    if len(fold_model_paths) == 0:
        print("[ERROR] No fold models were saved! Cannot run ensemble prediction.")
        print("  Possible causes:")
        print("  - Training data too small for the number of folds")
        print("  - Validation sets have only one class (AUROC undefined)")
        print(f"  Suggestion: reduce --n_folds (currently {args.n_folds}) or use more training data")
        return

    print(f"[Ensemble] Using {len(fold_model_paths)} fold model(s):")
    for fold_id, path in fold_model_paths:
        print(f"  Fold {fold_id}: {path}")

    df_all = pd.read_csv(folds_csv)
    threshold = 0.5
    print("[Predict] Using fixed threshold=0.5 for pred_label. Use pred_score for custom thresholding.")

    pred_score, pred_label, per_fold_outputs = predict_all(
        fold_model_paths, df_all, args.img_dir, args, device, threshold=threshold
    )


    prediction_stem = Path(args.data_csv).stem
    for fold_output in per_fold_outputs:
        fold_id = fold_output["fold"]
        fold_df = df_all.copy()
        fold_df["pred_score"] = fold_output["pred_score"]
        fold_df["pred_label"] = fold_output["pred_label"]
        fold_df["source_model"] = f"fold{fold_id}"
        fold_csv = output_dir / f"{prediction_stem}_predictions_fold{fold_id}.csv"
        fold_df.to_csv(fold_csv, index=False)
        print(f"[Final] Fold {fold_id} predictions saved -> {fold_csv}")


    df_all["pred_score"] = pred_score
    df_all["pred_label"] = pred_label
    df_all["source_model"] = "ensemble"

    output_csv = output_dir / f"{prediction_stem}_predictions_ensemble.csv"
    df_all.to_csv(output_csv, index=False)
    print(f"\n[Final] Predictions saved → {output_csv}")
    print(f"[Final] Columns: {list(df_all.columns)}")
    print(f"[Final] pred_score stats: mean={pred_score.mean():.4f} std={pred_score.std():.4f} "
          f"min={pred_score.min():.4f} max={pred_score.max():.4f}")
    print(f"[Final] pred_label distribution:\n{df_all['pred_label'].value_counts()}")

    if args.label in df_all.columns:
        test_mask = df_all["fold"] == -1
        test_df = df_all[test_mask]
        if len(test_df) > 0:
            metrics = all_classification_metrics(test_df[args.label].values, test_df["pred_score"].values)
            print(f"\n[Final] Test set metrics:")
            print(f"  AUROC: {metrics['AUROC']:.4f}")
            print(f"  AUPRC: {metrics['AUPRC']:.4f}")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
