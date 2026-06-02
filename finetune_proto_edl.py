"""
Prototype + Evidential Deep Learning (EDL) fine-tuning script.

This script reuses the Mammo-FM training pipeline while replacing the simple
EDL classifier with a class-wise prototype evidential head.
"""

import sys

if "--help" in sys.argv or "-h" in sys.argv:
    import argparse

    _p = argparse.ArgumentParser(
        description="5-Fold CV fine-tuning with Prototype + EDL for breast cancer detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _p.add_argument("--data-dir", default=r"G:\data", type=str)
    _p.add_argument("--img-dir", default="images_png", type=str)
    _p.add_argument("--csv-file", default="train_with_test_data.csv", type=str)
    _p.add_argument("--clip_chk_pt_path", default="./model/Mammo-FM_BatmanlabTrained_CLIP.tar", type=str)
    _p.add_argument("--model-save-dir", default="./best_model", type=str)
    _p.add_argument("--csv-output-dir", default="./output", type=str)
    _p.add_argument("--output-dir", default=None, type=str)
    _p.add_argument("--fold-csv", default=None, type=str)
    _p.add_argument("--use-existing-fold-csv", default="n", type=str)
    _p.add_argument("--overlap-policy", default="test", choices=["error", "test", "train", "training"])
    _p.add_argument("--split-by-cohort", default="y", type=str)
    _p.add_argument("--cohort-col", default="cohort_num", type=str)
    _p.add_argument("--train-cohorts", default="1-8", type=str)
    _p.add_argument("--test-cohorts", default="9-10", type=str)
    _p.add_argument("--dataset", default="Custom", type=str)
    _p.add_argument("--data_frac", default="1.0", type=str)
    _p.add_argument("--label", default="cancer", type=str)
    _p.add_argument(
        "--arch",
        default="breast_clip_det_b5_period_n_ft",
        choices=["breast_clip_det_b5_period_n_lp", "breast_clip_det_b5_period_n_ft"],
    )
    _p.add_argument("--freeze-backbone", default="n", choices=["y", "n"])
    _p.add_argument("--n_folds", default=5, type=int)
    _p.add_argument("--kfold0-val-frac", default=0.2, type=float)
    _p.add_argument("--kfold0-val-max-frac", default=0.5, type=float)
    _p.add_argument("--epochs", default=10, type=int)
    _p.add_argument("--early-stop", default=5, type=int)
    _p.add_argument("--batch-size", default=4, type=int)
    _p.add_argument("--micro-batch-size", default=1, type=int)
    _p.add_argument("--lr", default=5e-5, type=float)
    _p.add_argument("--weight-decay", default=1e-4, type=float)
    _p.add_argument("--weighted_BCE", "--weighted-bce", dest="weighted_BCE", default="y", type=str)
    _p.add_argument("--warmup-epochs", default=1, type=float)
    _p.add_argument("--img-size", nargs="+", default=[1520, 912], type=int)
    _p.add_argument("--seed", default=42, type=int)
    _p.add_argument("--num-workers", default=2, type=int)
    _p.add_argument("--gpu-id", default=0, type=int)
    _p.add_argument("--device", default="cuda", type=str)
    _p.add_argument("--apex", default="y", type=str)
    _p.add_argument("--print-freq", default=50, type=int)
    _p.add_argument("--log-freq", default=200, type=int)
    _p.add_argument("--alpha", default=10, type=float)
    _p.add_argument("--sigma", default=15, type=float)
    _p.add_argument("--p", default=1.0, type=float)
    _p.add_argument("--mean", default=0.3089279, type=float)
    _p.add_argument("--std", default=0.25053555408335154, type=float)
    _p.add_argument("--evidence-type", default="softplus", choices=["relu", "exp", "softplus"])
    _p.add_argument("--edl-loss-type", default="log", choices=["log", "digamma", "mse"])
    _p.add_argument("--annealing-coef", default=0.1, type=float)
    _p.add_argument("--edl-kl-weight", default=None, type=float)
    _p.add_argument("--annealing-step", default=None, type=float)
    _p.add_argument("--annealing-start-frac", default=0.0, type=float)
    _p.add_argument("--edl-proto-k", default=4, type=int)
    _p.add_argument("--edl-proto-topk", default=3, type=int)
    _p.add_argument("--edl-proto-temperature", default=1.0, type=float)
    _p.add_argument("--edl-proto-normalize", default="y", type=str)
    _p.add_argument("--edl-proto-class-weight", default=1.0, type=float)
    _p.add_argument("--edl-proto-attract-weight", default=0.1, type=float)
    _p.add_argument("--edl-proto-separation-weight", default=0.1, type=float)
    _p.add_argument("--edl-proto-diversity-weight", default=0.01, type=float)
    _p.add_argument("--edl-proto-loss-weight", "--edl_proto_loss_weight", dest="edl_proto_loss_weight", default=1.0, type=float)
    _p.add_argument("--edl-proto-margin", default=1.0, type=float)
    _p.add_argument("--edl-proto-balance-classes", default="y", type=str)
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")

from edl_loss import kl_divergence
from finetune_edl import (
    read_mammo_csv,
    resolve_project_path,
    CustomMammoDataset,
    collate_fn,
    get_train_transform,
    get_val_transform,
    EarlyStopping,
    parse_bool,
    normalize_fraction_arg,
    create_folds,
    labels_to_onehot,
    compute_fold_class_weights,
    save_loss_curve,
    save_all_folds_loss_curve,
    save_fold_split_csv,
    save_metrics_csv,
    get_edl_annealing_value,
    is_edl_annealing_complete,
    all_classification_metrics,
    seed_all,
    AverageMeter,
    timeSince,
    LinearWarmupCosineAnnealingLR,
)
from prototype_edl_model import MammoPrototypeEDLModel


class PrototypeEDLLoss(nn.Module):
    """EDL loss that consumes alpha produced by the prototype head."""

    def __init__(
        self,
        num_classes=2,
        total_epochs=10,
        annealing_start_frac=0.0,
        annealing_coef=1.0,
        annealing_step=None,
        loss_type="log",
        class_weights=None,
        class_loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.total_epochs = int(total_epochs)
        self.annealing_start_frac = float(annealing_start_frac)
        self.annealing_coef = float(annealing_coef)
        self.annealing_step = None if annealing_step in (None, "") else float(annealing_step)
        self.loss_type = str(loss_type).lower()
        self.class_loss_weight = float(class_loss_weight)
        self.current_epoch = 0
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.numel()} does not match num_classes={self.num_classes}"
                )
        self.class_weights = class_weights

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _annealing_value(self):
        return get_edl_annealing_value(
            self.current_epoch,
            self.total_epochs,
            annealing_step=self.annealing_step,
            annealing_start_frac=self.annealing_start_frac,
        )

    def forward(self, alpha, targets_onehot):
        S = torch.sum(alpha, dim=1, keepdim=True)
        probability = alpha / S

        if self.loss_type == "log":
            loss_1 = torch.sum(
                targets_onehot * (torch.log(S + 1e-10) - torch.log(alpha + 1e-10)),
                dim=1,
                keepdim=True,
            )
        elif self.loss_type == "digamma":
            loss_1 = torch.sum(
                targets_onehot * (torch.digamma(S) - torch.digamma(alpha)),
                dim=1,
                keepdim=True,
            )
        elif self.loss_type == "mse":
            p = alpha / S
            loss_err = torch.sum((targets_onehot - p) ** 2, dim=1, keepdim=True)
            loss_var = torch.sum(p * (1 - p) / (S + 1), dim=1, keepdim=True)
            loss_1 = loss_err + loss_var
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        target_indices = torch.argmax(targets_onehot, dim=1).long()
        if self.class_weights is not None:
            sample_weights = self.class_weights.to(
                device=loss_1.device,
                dtype=loss_1.dtype,
            )[target_indices].view(-1, 1)
            loss_1 = loss_1 * sample_weights

        class_log_prob = torch.log(probability + 1e-10)
        class_loss = F.nll_loss(
            class_log_prob,
            target_indices,
            weight=None if self.class_weights is None else self.class_weights.to(device=alpha.device, dtype=alpha.dtype),
            reduction="none",
        ).view(-1, 1)

        lambda_kl = self.annealing_coef * self._annealing_value()
        kl = kl_divergence(alpha, targets_onehot)
        edl_loss = (loss_1 + lambda_kl * kl).mean()
        class_loss = class_loss.mean()
        total_loss = edl_loss + self.class_loss_weight * class_loss

        return {
            "total_loss": total_loss,
            "edl_loss": edl_loss,
            "class_loss": class_loss,
            "annealing_value": float(self._annealing_value()),
            "kl_weight": float(lambda_kl),
        }


class PrototypeRegularizationLoss(nn.Module):
    def __init__(
        self,
        attract_weight=0.1,
        separation_weight=0.1,
        diversity_weight=0.01,
        margin=1.0,
        balance_classes=True,
    ):
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
        if proto_distances.ndim != 3:
            raise ValueError(f"Expected prototype_distances [B, C, K], got {tuple(proto_distances.shape)}")

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
        raw_total_loss = (
            self.attract_weight * attract_loss
            + self.separation_weight * separation_loss
            + self.diversity_weight * diversity_loss
        )

        return {
            "total_loss": raw_total_loss,
            "raw_total_loss": raw_total_loss,
            "attract_loss": attract_loss,
            "separation_loss": separation_loss,
            "diversity_loss": diversity_loss,
        }


def save_proto_component_curve(history_df, output_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    component_columns = [
        "train_total_loss",
        "train_edl_loss",
        "train_class_loss",
        "train_proto_reg_loss",
        "train_proto_reg_loss_raw",
        "train_proto_attract_loss",
        "train_proto_separation_loss",
        "train_proto_diversity_loss",
        "val_loss",
    ]
    available_columns = [col for col in component_columns if col in history_df.columns]
    if not available_columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for column in available_columns:
        ax.plot(history_df["epoch"], history_df[column], marker="o", linewidth=2, label=column)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_all_proto_component_curves(history_df, output_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    component_columns = [
        "train_total_loss",
        "train_edl_loss",
        "train_class_loss",
        "train_proto_reg_loss",
        "train_proto_reg_loss_raw",
        "train_proto_attract_loss",
        "train_proto_separation_loss",
        "train_proto_diversity_loss",
        "val_loss",
    ]
    available_columns = [col for col in component_columns if col in history_df.columns]
    if not available_columns:
        return

    folds = sorted(history_df["fold"].unique().tolist())
    ncols = 2 if len(folds) > 1 else 1
    nrows = math.ceil(len(folds) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, fold in zip(axes_flat, folds):
        fold_df = history_df[history_df["fold"] == fold].sort_values("epoch")
        for column in available_columns:
            ax.plot(fold_df["epoch"], fold_df[column], marker="o", linewidth=2, label=column)
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


def format_proto_weight_tag(weight):
    value = float(weight)
    if abs(value - round(value)) < 1e-8:
        return f"{value:.1f}"
    return f"{value:.4g}"


def append_dir_suffix(path_obj, suffix):
    path_obj = Path(path_obj)
    return path_obj.with_name(f"{path_obj.name}_{suffix}")


def extract_proto_topk(model_outputs, topk):
    proto_evidence = model_outputs["prototype_evidence"]
    proto_similarity = model_outputs["prototype_similarities"]
    actual_topk = min(int(topk), int(proto_evidence.size(-1)))
    topk_evidence, topk_idx = torch.topk(proto_evidence, k=actual_topk, dim=-1)
    topk_similarity = torch.gather(proto_similarity, dim=-1, index=topk_idx)
    return {
        "proto_topk_idx": topk_idx,
        "proto_topk_evidence": topk_evidence,
        "proto_topk_similarity": topk_similarity,
        "actual_topk": actual_topk,
    }


@torch.no_grad()
def collect_training_embeddings(model, train_df, img_dir, args, device):
    dataset = CustomMammoDataset(
        train_df,
        img_dir,
        label_col=args.label,
        transform=get_val_transform(tuple(args.img_size)),
        mean=args.mean,
        std=args.std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    embeddings = []
    labels = []
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    model.eval()

    for data in tqdm(loader, desc="[Proto Init] collect embeddings"):
        inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
        labels_batch = data["y"]
        bs = inputs.size(0)
        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                z = model.encode_image(inputs[mb_start:mb_end])
            if model.proto_head.normalize_embeddings:
                z = F.normalize(z, p=2, dim=-1, eps=1e-12)
            embeddings.append(z.detach().cpu())
            labels.append(labels_batch[mb_start:mb_end].detach().cpu())

    if not embeddings:
        raise ValueError("Prototype initialization loader produced no embeddings.")

    embedding_tensor = torch.cat(embeddings, dim=0)
    label_tensor = torch.cat(labels, dim=0).long()
    return embedding_tensor.numpy(), label_tensor.numpy()


def _fit_class_kmeans(class_embeddings, num_prototypes, seed, class_name):
    num_samples = int(class_embeddings.shape[0])
    if num_samples <= 0:
        raise ValueError(f"Cannot initialize prototypes: class {class_name} has 0 training samples.")

    n_clusters = min(int(num_prototypes), num_samples)
    if n_clusters == 1:
        centers = class_embeddings[:1].copy()
    else:
        estimator = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10)
        estimator.fit(class_embeddings)
        centers = estimator.cluster_centers_

    if n_clusters < int(num_prototypes):
        repeat_count = int(np.ceil(float(num_prototypes) / float(n_clusters)))
        centers = np.tile(centers, (repeat_count, 1))[: int(num_prototypes)]
        print(
            f"[Proto Init] WARNING: class {class_name} has only {num_samples} samples < K={num_prototypes}. "
            "Repeating centers to fill prototype slots."
        )
    return centers.astype(np.float32, copy=False)


def initialize_fold_prototypes(model, train_df, img_dir, args, device, fold_id):
    embeddings, labels = collect_training_embeddings(model, train_df, img_dir, args, device)
    num_prototypes = int(args.edl_proto_k)

    neg_embeddings = embeddings[labels == 0]
    pos_embeddings = embeddings[labels == 1]
    neg_centers = _fit_class_kmeans(neg_embeddings, num_prototypes, args.seed + int(fold_id), "0")
    pos_centers = _fit_class_kmeans(pos_embeddings, num_prototypes, args.seed + int(fold_id) + 997, "1")

    proto_tensor = torch.from_numpy(np.stack([neg_centers, pos_centers], axis=0))
    if model.proto_head.normalize_embeddings:
        proto_tensor = F.normalize(proto_tensor, p=2, dim=-1, eps=1e-12)

    model.initialize_prototypes(proto_tensor)
    print(
        f"[Proto Init] fold={fold_id} initialized prototypes with shape={tuple(proto_tensor.shape)} "
        f"from train embeddings ({len(neg_embeddings)} neg / {len(pos_embeddings)} pos)."
    )


def train_epoch_proto(model, loader, criterion, optimizer, scheduler, scaler, epoch, total_epochs, args, logger, device):
    model.train()
    total_losses = AverageMeter()
    edl_losses = AverageMeter()
    class_losses = AverageMeter()
    proto_reg_losses = AverageMeter()
    proto_reg_raw_losses = AverageMeter()
    proto_attract_losses = AverageMeter()
    proto_separation_losses = AverageMeter()
    proto_diversity_losses = AverageMeter()
    start = time.time()
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    num_classes = getattr(args, "num_classes", 2)

    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    for step, data in enumerate(tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs} train]")):
        inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
        labels = data["y"].to(device, non_blocking=True)
        bs = inputs.size(0)
        labels_onehot = labels_to_onehot(labels, num_classes=num_classes, device=device)

        optimizer.zero_grad(set_to_none=True)
        for mb_start in range(0, bs, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, bs)
            mb_inputs = inputs[mb_start:mb_end]
            mb_labels_onehot = labels_onehot[mb_start:mb_end]
            mb_size = mb_inputs.size(0)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(mb_inputs)
                loss_terms = criterion(outputs["alpha"], mb_labels_onehot)
                target_indices = torch.argmax(mb_labels_onehot, dim=1).long()
                proto_terms = args.prototype_regularizer(outputs, model.proto_head, target_indices)
                proto_loss_raw = proto_terms.get("raw_total_loss", proto_terms["total_loss"])
                proto_loss = args.edl_proto_loss_weight * proto_loss_raw
                total_loss = loss_terms["total_loss"] + proto_loss
                scaled_loss = total_loss * (mb_size / bs)

            total_losses.update(total_loss.item(), mb_size)
            edl_losses.update(loss_terms["edl_loss"].item(), mb_size)
            class_losses.update(loss_terms["class_loss"].item(), mb_size)
            proto_reg_losses.update(proto_loss.item(), mb_size)
            proto_reg_raw_losses.update(proto_loss_raw.item(), mb_size)
            proto_attract_losses.update(proto_terms["attract_loss"].item(), mb_size)
            proto_separation_losses.update(proto_terms["separation_loss"].item(), mb_size)
            proto_diversity_losses.update(proto_terms["diversity_loss"].item(), mb_size)
            scaler.scale(scaled_loss).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.print_freq == 0 or step == len(loader) - 1:
            print(
                f"Epoch [{epoch+1}][{step}/{len(loader)}] "
                f"Loss: {total_losses.val:.4f}({total_losses.avg:.4f}) "
                f"proto: {proto_reg_losses.val:.4f}({proto_reg_losses.avg:.4f}) "
                f"proto_raw: {proto_reg_raw_losses.val:.4f}({proto_reg_raw_losses.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                f"Remain: {timeSince(start, float(step+1)/len(loader))}"
            )

        if step % args.log_freq == 0 or step == len(loader) - 1:
            idx = step + len(loader) * epoch
            logger.add_scalar("train/iter_loss", total_losses.avg, idx)
            logger.add_scalar("train/iter_total_loss", total_losses.avg, idx)
            logger.add_scalar("train/iter_edl_loss", edl_losses.avg, idx)
            logger.add_scalar("train/iter_class_loss", class_losses.avg, idx)
            logger.add_scalar("train/iter_proto_reg_loss", proto_reg_losses.avg, idx)
            logger.add_scalar("train/iter_proto_reg_loss_raw", proto_reg_raw_losses.avg, idx)
            logger.add_scalar("train/iter_proto_attract_loss", proto_attract_losses.avg, idx)
            logger.add_scalar("train/iter_proto_separation_loss", proto_separation_losses.avg, idx)
            logger.add_scalar("train/iter_proto_diversity_loss", proto_diversity_losses.avg, idx)
            logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], idx)

    return {
        "total_loss": total_losses.avg,
        "edl_loss": edl_losses.avg,
        "class_loss": class_losses.avg,
        "proto_reg_loss": proto_reg_losses.avg,
        "proto_reg_loss_raw": proto_reg_raw_losses.avg,
        "proto_attract_loss": proto_attract_losses.avg,
        "proto_separation_loss": proto_separation_losses.avg,
        "proto_diversity_loss": proto_diversity_losses.avg,
    }


@torch.no_grad()
def valid_epoch_proto(model, loader, criterion, epoch, total_epochs, args, device):
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_evidence = []
    all_alpha = []
    all_uncertainty = []
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    num_classes = getattr(args, "num_classes", 2)

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
                outputs = model(mb_inputs)
                loss_terms = criterion(outputs["alpha"], mb_labels_onehot)
                loss = loss_terms["total_loss"]

            losses.update(loss.item(), mb_size)
            all_probs.append(outputs["probability"].detach().cpu().numpy())
            all_evidence.append(outputs["evidence"].detach().cpu().numpy())
            all_alpha.append(outputs["alpha"].detach().cpu().numpy())
            all_uncertainty.append(outputs["uncertainty"].detach().cpu().numpy())

    if not all_probs:
        raise ValueError("Validation loader produced no batches.")

    return (
        losses.avg,
        np.concatenate(all_probs),
        np.concatenate(all_evidence),
        np.concatenate(all_alpha),
        np.concatenate(all_uncertainty),
    )


@torch.no_grad()
def predict_all_proto(model_paths, df_all, img_dir, args, device, threshold=None):
    base_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if base_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["model_type"]

    dataset = CustomMammoDataset(
        df_all,
        img_dir,
        label_col=args.label,
        transform=None,
        mean=args.mean,
        std=args.std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    num_classes = getattr(args, "num_classes", 2)
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", args.batch_size)))
    amp_enabled = bool(args.apex and device.type == "cuda")
    topk = int(args.edl_proto_topk)
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
        model = MammoPrototypeEDLModel(
            args,
            ckpt=base_ckpt,
            num_classes=num_classes,
            evidence_type=args.evidence_type,
            prototypes_per_class=args.edl_proto_k,
            temperature=args.edl_proto_temperature,
            normalize_embeddings=args.edl_proto_normalize,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model = model.to(device)
        model.eval()

        fold_probs = []
        fold_evidence = []
        fold_alpha = []
        fold_uncertainty = []
        fold_topk_idx = []
        fold_topk_evidence = []
        fold_topk_similarity = []
        actual_topk = None

        for data in tqdm(loader, desc=f"[Predict] fold {fold_idx}"):
            inputs = data["x"].to(device, non_blocking=True).squeeze(1).permute(0, 3, 1, 2).contiguous()
            bs = inputs.size(0)
            for mb_start in range(0, bs, micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, bs)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(inputs[mb_start:mb_end])

                topk_outputs = extract_proto_topk(outputs, topk)
                actual_topk = topk_outputs["actual_topk"]
                fold_probs.append(outputs["probability"].detach().cpu().numpy())
                fold_evidence.append(outputs["evidence"].detach().cpu().numpy())
                fold_alpha.append(outputs["alpha"].detach().cpu().numpy())
                fold_uncertainty.append(outputs["uncertainty"].detach().cpu().numpy())
                fold_topk_idx.append(topk_outputs["proto_topk_idx"].detach().cpu().numpy())
                fold_topk_evidence.append(topk_outputs["proto_topk_evidence"].detach().cpu().numpy())
                fold_topk_similarity.append(topk_outputs["proto_topk_similarity"].detach().cpu().numpy())

        fold_probability = np.concatenate(fold_probs)
        fold_evidence_arr = np.concatenate(fold_evidence)
        fold_alpha_arr = np.concatenate(fold_alpha)
        fold_uncertainty_arr = np.concatenate(fold_uncertainty)
        fold_topk_idx_arr = np.concatenate(fold_topk_idx)
        fold_topk_evidence_arr = np.concatenate(fold_topk_evidence)
        fold_topk_similarity_arr = np.concatenate(fold_topk_similarity)

        fold_score = fold_probability[:, 1]
        fold_label = (fold_score >= th).astype(int)

        all_scores.append(fold_score)
        all_probabilities.append(fold_probability)
        all_evidences.append(fold_evidence_arr)
        all_alphas.append(fold_alpha_arr)
        all_uncertainties.append(fold_uncertainty_arr)
        per_fold_results.append(
            {
                "fold": int(fold_idx),
                "pred_score": fold_score,
                "pred_label": fold_label,
                "probability": fold_probability,
                "evidence": fold_evidence_arr,
                "alpha": fold_alpha_arr,
                "uncertainty": fold_uncertainty_arr,
                "proto_topk_idx": fold_topk_idx_arr,
                "proto_topk_evidence": fold_topk_evidence_arr,
                "proto_topk_similarity": fold_topk_similarity_arr,
                "proto_topk_source_fold": np.full(
                    fold_topk_idx_arr.shape,
                    int(fold_idx),
                    dtype=np.int64,
                ),
                "actual_topk": int(actual_topk),
            }
        )

        torch.cuda.empty_cache()

    pred_score = np.mean(all_scores, axis=0)
    pred_probability = np.mean(all_probabilities, axis=0)
    pred_evidence = np.mean(all_evidences, axis=0)
    pred_alpha = np.mean(all_alphas, axis=0)
    pred_uncertainty = np.mean(all_uncertainties, axis=0)
    pred_label = (pred_score >= th).astype(int)

    num_samples = pred_score.shape[0]
    actual_topk = int(per_fold_results[0]["actual_topk"])
    ensemble_topk_idx = np.zeros((num_samples, num_classes, actual_topk), dtype=np.int64)
    ensemble_topk_evidence = np.zeros((num_samples, num_classes, actual_topk), dtype=np.float32)
    ensemble_topk_similarity = np.zeros((num_samples, num_classes, actual_topk), dtype=np.float32)
    ensemble_topk_source_fold = np.zeros((num_samples, num_classes, actual_topk), dtype=np.int64)

    fold_ids = np.array([item["fold"] for item in per_fold_results], dtype=np.int64)
    fold_class_evidence = np.stack([item["evidence"] for item in per_fold_results], axis=0)
    for class_idx in range(num_classes):
        source_pos = np.argmax(fold_class_evidence[:, :, class_idx], axis=0)
        for sample_idx, fold_pos in enumerate(source_pos.tolist()):
            ensemble_topk_idx[sample_idx, class_idx, :] = per_fold_results[fold_pos]["proto_topk_idx"][
                sample_idx, class_idx, :
            ]
            ensemble_topk_evidence[sample_idx, class_idx, :] = per_fold_results[fold_pos]["proto_topk_evidence"][
                sample_idx, class_idx, :
            ]
            ensemble_topk_similarity[sample_idx, class_idx, :] = per_fold_results[fold_pos]["proto_topk_similarity"][
                sample_idx, class_idx, :
            ]
            ensemble_topk_source_fold[sample_idx, class_idx, :] = fold_ids[fold_pos]

    return (
        {
            "pred_score": pred_score,
            "pred_label": pred_label,
            "probability": pred_probability,
            "evidence": pred_evidence,
            "alpha": pred_alpha,
            "uncertainty": pred_uncertainty,
            "proto_topk_idx": ensemble_topk_idx,
            "proto_topk_evidence": ensemble_topk_evidence,
            "proto_topk_similarity": ensemble_topk_similarity,
            "proto_topk_source_fold": ensemble_topk_source_fold,
            "actual_topk": actual_topk,
        },
        per_fold_results,
    )


def save_prediction_csv(df_all, results, output_path, fold_idx=None):
    df_out = df_all.copy()
    df_out["pred_score"] = results["pred_score"]
    df_out["pred_label"] = results["pred_label"]
    df_out["prediction_score"] = results["pred_score"]
    df_out["predicted_class"] = results["pred_label"]

    num_classes = results["evidence"].shape[1]
    for class_idx in range(num_classes):
        df_out[f"evidence_{class_idx}"] = results["evidence"][:, class_idx]
        df_out[f"alpha_{class_idx}"] = results["alpha"][:, class_idx]
        df_out[f"probability_{class_idx}"] = results["probability"][:, class_idx]

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


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="5-Fold CV fine-tuning with Prototype + EDL for breast cancer detection",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument("--data-dir", default=r"G:\data", type=str)
        parser.add_argument("--img-dir", default="images_png", type=str)
        parser.add_argument("--csv-file", default="train_with_test_data.csv", type=str)
        parser.add_argument("--clip_chk_pt_path", default="./model/Mammo-FM_BatmanlabTrained_CLIP.tar", type=str)
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
        parser.add_argument("--dataset", default="Custom", type=str)
        parser.add_argument("--data_frac", default="1.0", type=str)
        parser.add_argument("--label", default="cancer", type=str)
        parser.add_argument(
            "--arch",
            default="breast_clip_det_b5_period_n_ft",
            choices=["breast_clip_det_b5_period_n_lp", "breast_clip_det_b5_period_n_ft"],
        )
        parser.add_argument("--freeze-backbone", default="n", choices=["y", "n"])
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
        parser.add_argument("--gpu-id", default=0, type=int)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--apex", default="y", type=str)
        parser.add_argument("--print-freq", default=50, type=int)
        parser.add_argument("--log-freq", default=200, type=int)
        parser.add_argument("--alpha", default=10, type=float)
        parser.add_argument("--sigma", default=15, type=float)
        parser.add_argument("--p", default=1.0, type=float)
        parser.add_argument("--mean", default=0.3089279, type=float)
        parser.add_argument("--std", default=0.25053555408335154, type=float)
        parser.add_argument("--evidence-type", default="softplus", choices=["relu", "exp", "softplus"])
        parser.add_argument("--edl-loss-type", default="log", choices=["log", "digamma", "mse"])
        parser.add_argument("--annealing-coef", default=0.1, type=float)
        parser.add_argument("--edl-kl-weight", default=None, type=float)
        parser.add_argument("--annealing-step", default=None, type=float)
        parser.add_argument("--annealing-start-frac", default=0.0, type=float)
        parser.add_argument("--edl-proto-k", default=4, type=int)
        parser.add_argument("--edl-proto-topk", default=3, type=int)
        parser.add_argument("--edl-proto-temperature", default=1.0, type=float)
        parser.add_argument("--edl-proto-normalize", default="y", type=str)
        parser.add_argument("--edl-proto-class-weight", default=1.0, type=float)
        parser.add_argument("--edl-proto-attract-weight", default=0.1, type=float)
        parser.add_argument("--edl-proto-separation-weight", default=0.1, type=float)
        parser.add_argument("--edl-proto-diversity-weight", default=0.01, type=float)
        parser.add_argument("--edl-proto-loss-weight", "--edl_proto_loss_weight", dest="edl_proto_loss_weight", default=1.0, type=float)
        parser.add_argument("--edl-proto-margin", default=1.0, type=float)
        parser.add_argument("--edl-proto-balance-classes", default="y", type=str)
        args = parser.parse_args()

    args.num_classes = 2
    freeze_backbone = getattr(args, "freeze_backbone", False)
    if isinstance(freeze_backbone, str):
        freeze_backbone = freeze_backbone.lower() in {"1", "true", "t", "yes", "y"}
    args.freeze_backbone = bool(freeze_backbone)
    args.evidence_type = str(getattr(args, "evidence_type", "softplus")).lower()
    args.edl_loss_type = str(getattr(args, "edl_loss_type", "log")).lower()
    args.annealing_coef = float(getattr(args, "annealing_coef", 0.1))
    if getattr(args, "edl_kl_weight", None) is not None:
        args.annealing_coef = float(args.edl_kl_weight)
    else:
        args.edl_kl_weight = float(args.annealing_coef)
    if getattr(args, "annealing_step", None) in ("",):
        args.annealing_step = None
    if getattr(args, "annealing_step", None) is not None:
        args.annealing_step = float(args.annealing_step)
        if args.annealing_step <= 0:
            raise ValueError(f"annealing_step must be > 0, got {args.annealing_step}")
    args.annealing_start_frac = float(getattr(args, "annealing_start_frac", 0.0))
    if not hasattr(args, "weighted_BCE"):
        args.weighted_BCE = getattr(args, "weighted_bce", "y")
    args.weighted_bce = str(args.weighted_BCE).lower() in {"1", "true", "t", "yes", "y"}
    args.use_existing_fold_csv = parse_bool(getattr(args, "use_existing_fold_csv", False), default=False)
    args.n_folds = int(args.n_folds)
    if args.n_folds < 0 or args.n_folds == 1:
        raise ValueError(f"n_folds must be 0 or >= 2, got {args.n_folds}")
    args.kfold0_val_frac = normalize_fraction_arg(getattr(args, "kfold0_val_frac", 0.2), "kfold0_val_frac")
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
    overlap_policy = {"raise": "error", "strict": "error", "training": "train"}.get(overlap_policy, overlap_policy)
    if overlap_policy not in {"error", "test", "train"}:
        raise ValueError(f"Unsupported overlap_policy={overlap_policy!r}. Expected error, test, or train.")
    args.overlap_policy = overlap_policy

    args.edl_proto_k = int(getattr(args, "edl_proto_k", 4))
    args.edl_proto_topk = int(getattr(args, "edl_proto_topk", 3))
    args.edl_proto_temperature = float(getattr(args, "edl_proto_temperature", 1.0))
    args.edl_proto_normalize = parse_bool(getattr(args, "edl_proto_normalize", True), default=True)
    args.edl_proto_class_weight = float(getattr(args, "edl_proto_class_weight", 1.0))
    args.edl_proto_attract_weight = float(getattr(args, "edl_proto_attract_weight", 0.1))
    args.edl_proto_separation_weight = float(getattr(args, "edl_proto_separation_weight", 0.1))
    args.edl_proto_diversity_weight = float(getattr(args, "edl_proto_diversity_weight", 0.01))
    args.edl_proto_loss_weight = float(getattr(args, "edl_proto_loss_weight", 1.0))
    args.edl_proto_margin = float(getattr(args, "edl_proto_margin", 1.0))
    args.edl_proto_balance_classes = parse_bool(getattr(args, "edl_proto_balance_classes", True), default=True)
    if args.edl_proto_k <= 0:
        raise ValueError(f"edl_proto_k must be > 0, got {args.edl_proto_k}")
    if args.edl_proto_topk <= 0:
        raise ValueError(f"edl_proto_topk must be > 0, got {args.edl_proto_topk}")
    if args.edl_proto_temperature <= 0:
        raise ValueError(f"edl_proto_temperature must be > 0, got {args.edl_proto_temperature}")
    if args.edl_proto_margin <= 0:
        raise ValueError(f"edl_proto_margin must be > 0, got {args.edl_proto_margin}")
    if args.edl_proto_loss_weight < 0:
        raise ValueError(f"edl_proto_loss_weight must be >= 0, got {args.edl_proto_loss_weight}")
    if args.edl_proto_topk > args.edl_proto_k:
        print(
            f"[Proto] WARNING: edl_proto_topk={args.edl_proto_topk} > edl_proto_k={args.edl_proto_k}. "
            f"Clamping topk to {args.edl_proto_k}."
        )
        args.edl_proto_topk = args.edl_proto_k

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
    if abs(args.edl_proto_loss_weight - 1.0) > 1e-8:
        proto_weight_tag = f"proto_w{format_proto_weight_tag(args.edl_proto_loss_weight)}"
        model_save_dir = append_dir_suffix(model_save_dir, proto_weight_tag)
        csv_output_dir = append_dir_suffix(csv_output_dir, proto_weight_tag)
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
    print("  Prototype + Evidential Deep Learning (EDL) Fine-tuning")
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
    print(f"EDL Loss Type: {args.edl_loss_type}")
    print(f"EDL KL Weight(lambda): {args.edl_kl_weight}")
    print(f"EDL Annealing Step: {args.annealing_step}")
    print(f"EDL Annealing Start Frac: {args.annealing_start_frac}")
    print(f"Weighted BCE/Data Loss: {args.weighted_bce}")
    print(f"Prototype K: {args.edl_proto_k}")
    print(f"Prototype TopK: {args.edl_proto_topk}")
    print(f"Prototype Temperature: {args.edl_proto_temperature}")
    print(f"Prototype Normalize: {args.edl_proto_normalize}")
    print(f"Prototype Class Loss Weight: {args.edl_proto_class_weight}")
    print(f"Prototype Attract Weight: {args.edl_proto_attract_weight}")
    print(f"Prototype Separation Weight: {args.edl_proto_separation_weight}")
    print(f"Prototype Diversity Weight: {args.edl_proto_diversity_weight}")
    print(f"Prototype Loss Weight: {args.edl_proto_loss_weight}")
    print(f"Prototype Margin: {args.edl_proto_margin}")
    print(f"Prototype Balance Classes: {args.edl_proto_balance_classes}")
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
        folds_csv = create_folds(
            args.data_csv,
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

    base_ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if base_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = base_ckpt["config"]["model"]["image_encoder"]["model_type"]

    fold_model_paths = []
    oof_parts = []
    all_fold_histories = []
    data_stem = Path(args.data_csv).stem
    fold_ids = [0] if args.n_folds == 0 else list(range(args.n_folds))
    args.prototype_regularizer = PrototypeRegularizationLoss(
        attract_weight=args.edl_proto_attract_weight,
        separation_weight=args.edl_proto_separation_weight,
        diversity_weight=args.edl_proto_diversity_weight,
        margin=args.edl_proto_margin,
        balance_classes=args.edl_proto_balance_classes,
    )

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

        train_ds = CustomMammoDataset(
            train_df,
            args.img_dir,
            label_col=args.label,
            transform=train_transform,
            mean=args.mean,
            std=args.std,
        )
        valid_ds = CustomMammoDataset(
            valid_df,
            args.img_dir,
            label_col=args.label,
            transform=val_transform,
            mean=args.mean,
            std=args.std,
        )

        if len(train_ds) == 0 or len(valid_ds) == 0:
            print(f"  WARNING: Fold {fold} has empty train or valid data; skipping.")
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        if len(train_loader) == 0 or len(valid_loader) == 0:
            print(f"  WARNING: Fold {fold} produced no train or valid batches; skipping.")
            continue

        model = MammoPrototypeEDLModel(
            args,
            ckpt=base_ckpt,
            num_classes=args.num_classes,
            evidence_type=args.evidence_type,
            prototypes_per_class=args.edl_proto_k,
            temperature=args.edl_proto_temperature,
            normalize_embeddings=args.edl_proto_normalize,
        )
        model = model.to(device)
        try:
            initialize_fold_prototypes(model, train_df, args.img_dir, args, device, fold)
        except Exception as exc:
            print(f"  ERROR: Fold {fold} prototype initialization failed: {exc}")
            import traceback

            traceback.print_exc()
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found. Check freeze_backbone/model setup.")
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        backbone_count = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
        head_count = sum(p.numel() for p in model.proto_head.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_count:,} / {total_count:,}")
        print(f"  Trainable backbone params: {backbone_count:,}")
        print(f"  Trainable prototype head params: {head_count:,}")

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        warmup_steps = len(train_loader) if args.warmup_epochs >= 1 else int(args.warmup_epochs * len(train_loader))
        lr_config = {
            "total_epochs": args.epochs,
            "warmup_steps": warmup_steps,
            "total_steps": len(train_loader) * args.epochs,
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex)

        class_weights = compute_fold_class_weights(train_df, args.label, args.weighted_bce)
        criterion = PrototypeEDLLoss(
            num_classes=args.num_classes,
            total_epochs=args.epochs,
            annealing_start_frac=args.annealing_start_frac,
            annealing_coef=args.annealing_coef,
            annealing_step=args.annealing_step,
            loss_type=args.edl_loss_type,
            class_weights=class_weights,
            class_loss_weight=args.edl_proto_class_weight,
        )

        logger = SummaryWriter(log_dir / f"fold{fold}")
        early_stopper = EarlyStopping(patience=args.early_stop) if args.early_stop > 0 else None
        best_auroc = -float("inf")
        best_model_path = ckpt_dir / f"best_fold{fold}_seed{args.seed}_proto_edl.pth"
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

                train_metrics = train_epoch_proto(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args.epochs,
                    args,
                    logger,
                    device,
                )

                val_loss, predictions, evidences, alphas, uncertainties = valid_epoch_proto(
                    model,
                    valid_loader,
                    criterion,
                    epoch,
                    args.epochs,
                    args,
                    device,
                )
                last_predictions = predictions.copy()
                last_evidences = evidences.copy()
                last_alphas = alphas.copy()
                last_uncertainties = uncertainties.copy()

                pred_score = predictions[:, 1]
                valid_df_fold = valid_df.copy()
                valid_df_fold["prediction"] = pred_score
                valid_agg = valid_df_fold.groupby("patient_id").agg(
                    {args.label: "max", "prediction": "max"}
                )
                metrics = all_classification_metrics(valid_agg[args.label].values, valid_agg["prediction"].values)
                last_metrics = metrics
                elapsed = time.time() - start
                mean_uncertainty = float(np.mean(uncertainties))
                annealing_value = get_edl_annealing_value(
                    epoch,
                    args.epochs,
                    annealing_step=args.annealing_step,
                    annealing_start_frac=args.annealing_start_frac,
                )
                annealing_complete = is_edl_annealing_complete(
                    epoch,
                    args.epochs,
                    annealing_step=args.annealing_step,
                    annealing_start_frac=args.annealing_start_frac,
                )
                is_best_epoch = metrics["AUROC"] > best_auroc
                fold_history_rows.append(
                    {
                        "fold": fold,
                        "epoch": epoch + 1,
                        "eval_split": eval_split_name,
                        "train_loss": float(train_metrics["total_loss"]),
                        "train_total_loss": float(train_metrics["total_loss"]),
                        "train_edl_loss": float(train_metrics["edl_loss"]),
                        "train_class_loss": float(train_metrics["class_loss"]),
                        "train_proto_reg_loss": float(train_metrics["proto_reg_loss"]),
                        "train_proto_reg_loss_raw": float(train_metrics["proto_reg_loss_raw"]),
                        "train_proto_attract_loss": float(train_metrics["proto_attract_loss"]),
                        "train_proto_separation_loss": float(train_metrics["proto_separation_loss"]),
                        "train_proto_diversity_loss": float(train_metrics["proto_diversity_loss"]),
                        "edl_proto_loss_weight": float(args.edl_proto_loss_weight),
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
                    }
                )

                print(
                    f"Fold {fold} Epoch {epoch+1} - train_loss: {train_metrics['total_loss']:.4f}  "
                    f"proto: {train_metrics['proto_reg_loss']:.4f}  "
                    f"proto_raw: {train_metrics['proto_reg_loss_raw']:.4f}  "
                    f"{eval_split_name}_loss: {val_loss:.4f}  AUROC: {metrics['AUROC']:.4f}  "
                    f"AUPRC: {metrics['AUPRC']:.4f}  bACC: {metrics['bACC']:.4f}  time: {elapsed:.0f}s"
                )

                logger.add_scalar("train/epoch_total_loss", train_metrics["total_loss"], epoch + 1)
                logger.add_scalar("train/epoch_edl_loss", train_metrics["edl_loss"], epoch + 1)
                logger.add_scalar("train/epoch_class_loss", train_metrics["class_loss"], epoch + 1)
                logger.add_scalar("train/epoch_proto_reg_loss", train_metrics["proto_reg_loss"], epoch + 1)
                logger.add_scalar("train/epoch_proto_reg_loss_raw", train_metrics["proto_reg_loss_raw"], epoch + 1)
                logger.add_scalar("train/epoch_proto_attract_loss", train_metrics["proto_attract_loss"], epoch + 1)
                logger.add_scalar("train/epoch_proto_separation_loss", train_metrics["proto_separation_loss"], epoch + 1)
                logger.add_scalar("train/epoch_proto_diversity_loss", train_metrics["proto_diversity_loss"], epoch + 1)
                logger.add_scalar("train/edl_proto_loss_weight", args.edl_proto_loss_weight, epoch + 1)
                logger.add_scalar("train/edl_annealing_value", annealing_value, epoch + 1)
                logger.add_scalar(f"{eval_split_name}/AUROC", metrics["AUROC"], epoch + 1)
                logger.add_scalar(f"{eval_split_name}/AUPRC", metrics["AUPRC"], epoch + 1)
                logger.add_scalar(f"{eval_split_name}/bACC", metrics["bACC"], epoch + 1)
                logger.add_scalar(f"{eval_split_name}/loss", val_loss, epoch + 1)
                logger.add_scalar(f"{eval_split_name}/mean_uncertainty", mean_uncertainty, epoch + 1)

                if is_best_epoch:
                    best_auroc = metrics["AUROC"]
                    best_predictions = predictions.copy()
                    best_evidences = evidences.copy()
                    best_alphas = alphas.copy()
                    best_uncertainties = uncertainties.copy()
                    best_metrics = metrics
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "auroc": best_auroc,
                            "metrics": metrics,
                            "prototype_config": {
                                "edl_proto_k": args.edl_proto_k,
                                "edl_proto_topk": args.edl_proto_topk,
                                "edl_proto_temperature": args.edl_proto_temperature,
                                "edl_proto_normalize": args.edl_proto_normalize,
                                "edl_proto_class_weight": args.edl_proto_class_weight,
                                "edl_proto_attract_weight": args.edl_proto_attract_weight,
                                "edl_proto_separation_weight": args.edl_proto_separation_weight,
                                "edl_proto_diversity_weight": args.edl_proto_diversity_weight,
                                "edl_proto_loss_weight": args.edl_proto_loss_weight,
                                "edl_proto_margin": args.edl_proto_margin,
                                "edl_proto_balance_classes": args.edl_proto_balance_classes,
                            },
                        },
                        best_model_path,
                    )
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
                torch.save(
                    {
                        "epoch": last_epoch,
                        "model": model.state_dict(),
                        "auroc": best_auroc,
                        "metrics": best_metrics,
                        "prototype_config": {
                            "edl_proto_k": args.edl_proto_k,
                            "edl_proto_topk": args.edl_proto_topk,
                            "edl_proto_temperature": args.edl_proto_temperature,
                            "edl_proto_normalize": args.edl_proto_normalize,
                            "edl_proto_class_weight": args.edl_proto_class_weight,
                            "edl_proto_attract_weight": args.edl_proto_attract_weight,
                            "edl_proto_separation_weight": args.edl_proto_separation_weight,
                            "edl_proto_diversity_weight": args.edl_proto_diversity_weight,
                            "edl_proto_loss_weight": args.edl_proto_loss_weight,
                            "edl_proto_margin": args.edl_proto_margin,
                            "edl_proto_balance_classes": args.edl_proto_balance_classes,
                        },
                    },
                    best_model_path,
                )

        except Exception as exc:
            print(f"  ERROR: Fold {fold} training failed with exception: {exc}")
            import traceback

            traceback.print_exc()
            if not saved_best:
                try:
                    torch.save(
                        {
                            "epoch": -1,
                            "model": model.state_dict(),
                            "auroc": -1.0,
                            "metrics": {},
                            "prototype_config": {
                                "edl_proto_k": args.edl_proto_k,
                                "edl_proto_topk": args.edl_proto_topk,
                                "edl_proto_temperature": args.edl_proto_temperature,
                                "edl_proto_normalize": args.edl_proto_normalize,
                                "edl_proto_class_weight": args.edl_proto_class_weight,
                                "edl_proto_attract_weight": args.edl_proto_attract_weight,
                                "edl_proto_separation_weight": args.edl_proto_separation_weight,
                                "edl_proto_diversity_weight": args.edl_proto_diversity_weight,
                                "edl_proto_loss_weight": args.edl_proto_loss_weight,
                                "edl_proto_margin": args.edl_proto_margin,
                                "edl_proto_balance_classes": args.edl_proto_balance_classes,
                            },
                        },
                        best_model_path,
                    )
                    print(f"  Saved emergency checkpoint for fold {fold}")
                except Exception as save_exc:
                    print(f"  Could not save emergency checkpoint: {save_exc}")

        if fold_history_rows:
            history_df = pd.DataFrame(fold_history_rows)
            history_csv = output_dir / f"{data_stem}_fold{fold}_loss_history_proto_edl.csv"
            history_png = output_dir / f"{data_stem}_fold{fold}_loss_curve_proto_edl.png"
            history_component_png = output_dir / f"{data_stem}_fold{fold}_loss_components_proto_edl.png"
            history_df.to_csv(history_csv, index=False)
            save_loss_curve(history_df, history_png, title=f"Fold {fold} Prototype EDL Loss Curve")
            save_proto_component_curve(
                history_df,
                history_component_png,
                title=f"Fold {fold} Prototype EDL Loss Components",
            )
            all_fold_histories.append(history_df)
            print(f"  Saved fold {fold} loss history -> {history_csv}")
            print(f"  Saved fold {fold} loss curve   -> {history_png}")
            print(f"  Saved fold {fold} component loss curve -> {history_component_png}")

        if best_model_path.exists():
            fold_model_paths.append((fold, best_model_path))
            if args.n_folds > 0 and best_predictions is not None:
                oof_part = valid_df.copy()
                oof_part["oof_pred_score"] = best_predictions[:, 1]
                for class_idx in range(args.num_classes):
                    oof_part[f"oof_probability_{class_idx}"] = best_predictions[:, class_idx]
                    oof_part[f"oof_evidence_{class_idx}"] = best_evidences[:, class_idx]
                    oof_part[f"oof_alpha_{class_idx}"] = best_alphas[:, class_idx]
                oof_part["oof_uncertainty"] = best_uncertainties.squeeze()
                oof_part["oof_fold"] = fold
                oof_parts.append(oof_part)
        else:
            print(f"  WARNING: No checkpoint for fold {fold}, skipping in ensemble.")

        logger.close()
        del model, optimizer, scheduler, scaler, criterion
        torch.cuda.empty_cache()
        gc.collect()

    if all_fold_histories:
        all_history_df = pd.concat(all_fold_histories, ignore_index=True)
        all_history_csv = output_dir / f"{data_stem}_all_folds_loss_history_proto_edl.csv"
        all_history_png = output_dir / f"{data_stem}_all_folds_loss_curve_proto_edl.png"
        all_history_component_png = output_dir / f"{data_stem}_all_folds_loss_components_proto_edl.png"
        all_history_df.to_csv(all_history_csv, index=False)
        save_all_folds_loss_curve(all_history_df, all_history_png, title="All Folds Prototype EDL Loss Curves")
        save_all_proto_component_curves(
            all_history_df,
            all_history_component_png,
            title="All Folds Prototype EDL Loss Components",
        )
        print(f"\n[Loss] All folds history saved -> {all_history_csv}")
        print(f"[Loss] All folds curves saved  -> {all_history_png}")
        print(f"[Loss] All folds component curves saved -> {all_history_component_png}")

    if oof_parts:
        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_csv = output_dir / f"{Path(args.data_csv).stem}_oof_proto_edl_predictions.csv"
        oof_df.to_csv(oof_csv, index=False)
        print(f"\n[OOF] Out-of-fold predictions saved -> {oof_csv}")

    print(f"\n{'=' * 60}")
    print("  Ensemble prediction on ALL data (Prototype + EDL)")
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
    ensemble_results, per_fold_results = predict_all_proto(
        fold_model_paths,
        df_all,
        args.img_dir,
        args,
        device,
        threshold=threshold,
    )

    prediction_stem = Path(args.data_csv).stem
    for fold_result in per_fold_results:
        fold_id = fold_result["fold"]
        fold_csv = output_dir / f"{prediction_stem}_predictions_proto_edl_fold{fold_id}.csv"
        save_prediction_csv(df_all, fold_result, fold_csv, fold_idx=fold_id)
        print(f"[Final] Fold {fold_id} Prototype EDL predictions saved -> {fold_csv}")

    ensemble_csv = output_dir / f"{prediction_stem}_predictions_proto_edl_ensemble.csv"
    ensemble_df = save_prediction_csv(df_all, ensemble_results, ensemble_csv, fold_idx=None)

    print(f"\n[Final] Ensemble Prototype EDL predictions saved -> {ensemble_csv}")
    print(f"[Final] Columns: {list(ensemble_df.columns)}")
    print(
        f"[Final] pred_score stats: mean={ensemble_results['pred_score'].mean():.4f} "
        f"std={ensemble_results['pred_score'].std():.4f} "
        f"min={ensemble_results['pred_score'].min():.4f} "
        f"max={ensemble_results['pred_score'].max():.4f}"
    )
    print(
        f"[Final] mean uncertainty: {ensemble_results['uncertainty'].mean():.4f} "
        f"std: {ensemble_results['uncertainty'].std():.4f}"
    )
    print(f"[Final] pred_label distribution:\n{ensemble_df['pred_label'].value_counts()}")

    if args.label in ensemble_df.columns:
        test_mask = ensemble_df["split"] == "test"
        test_df = ensemble_df[test_mask]
        if len(test_df) > 0:
            metrics = all_classification_metrics(test_df[args.label].values, test_df["pred_score"].values)
            metrics = dict(metrics)
            metrics["edl_proto_loss_weight"] = float(args.edl_proto_loss_weight)
            print("\n[Final] Test set metrics:")
            print(f"  AUROC: {metrics['AUROC']:.4f}")
            print(f"  AUPRC: {metrics['AUPRC']:.4f}")
            print(f"  bACC:  {metrics['bACC']:.4f}")
            print(f"  Mean uncertainty (test): {test_df['uncertainty'].mean():.4f}")
            metrics_csv = output_dir / f"{prediction_stem}_test_metrics_proto_edl.csv"
            save_metrics_csv(metrics, metrics_csv, split_name="test")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
