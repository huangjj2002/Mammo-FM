"""
Evidential Deep Learning (EDL) 损失函数模块

基于 Dirichlet 分布的证据深度学习损失函数，包含：
1. 类交叉熵损失 (Type II Maximum Likelihood)
2. KL 散度正则化项
3. 综合损失函数

参考: Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
"""

import torch
import torch.nn.functional as F


def relu_evidence(logits):
    """使用 ReLU 将 logits 转换为非负 evidence"""
    return F.relu(logits)


def exp_evidence(logits):
    """使用 exp 将 logits 转换为非负 evidence（更平滑）"""
    return torch.exp(torch.clamp(logits, min=-10, max=10))


def softplus_evidence(logits):
    """使用 softplus 将 logits 转换为非负 evidence"""
    return F.softplus(logits)


def get_evidence(logits, evidence_type="softplus"):
    """根据类型获取 evidence

    Args:
        logits: 模型原始输出 [batch_size, num_classes]
        evidence_type: 'relu', 'exp', 'softplus'

    Returns:
        evidence: 非负 evidence [batch_size, num_classes]
    """
    evidence_type = str(evidence_type).lower()
    if evidence_type == "relu":
        return relu_evidence(logits)
    elif evidence_type == "exp":
        return exp_evidence(logits)
    elif evidence_type == "softplus":
        return softplus_evidence(logits)
    else:
        raise ValueError(f"Unknown evidence_type: {evidence_type}")


def _annealed_kl_weight(epoch_num, annealing_start=1.0, annealing_coef=1.0):
    """Return a bounded KL weight for the legacy EDL loss helpers."""
    if annealing_start <= 0:
        return float(annealing_coef)
    return float(annealing_coef) * min(1.0, float(epoch_num) / float(annealing_start))


def edl_mse_loss(alpha, target, epoch_num, annealing_start=0.01, annealing_coef=1.0):
    """EDL MSE 损失（均方误差变体）

    Args:
        alpha: Dirichlet 参数 [batch_size, num_classes]
        target: one-hot 标签 [batch_size, num_classes]
        epoch_num: 当前 epoch 编号（用于退火）
        annealing_start: 退火系数
        annealing_coef: 退火系数缩放
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    loss_err = torch.sum((target - p) ** 2, dim=1, keepdim=True)
    loss_var = torch.sum(p * (1 - p) / (S + 1), dim=1, keepdim=True)
    annealing_coef_val = _annealed_kl_weight(epoch_num, annealing_start, annealing_coef)
    loss = loss_err + loss_var + annealing_coef_val * kl_divergence(alpha, target)
    return loss


def edl_digamma_loss(alpha, target, epoch_num, annealing_start=0.01, annealing_coef=1.0):
    """EDL Digamma 损失

    Args:
        alpha: Dirichlet 参数 [batch_size, num_classes]
        target: one-hot 标签 [batch_size, num_classes]
        epoch_num: 当前 epoch 编号
        annealing_start: 退火起始比例
        annealing_coef: 退火系数
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    loss_1 = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef_val = _annealed_kl_weight(epoch_num, annealing_start, annealing_coef)
    loss = loss_1 + annealing_coef_val * kl_divergence(alpha, target)
    return loss


def edl_log_loss(alpha, target, epoch_num, annealing_start=0.01, annealing_coef=1.0):
    """EDL Log 损失

    Args:
        alpha: Dirichlet 参数 [batch_size, num_classes]
        target: one-hot 标签 [batch_size, num_classes]
        epoch_num: 当前 epoch 编号
        annealing_start: 退火起始比例
        annealing_coef: 退火系数
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    loss_1 = torch.sum(target * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
    annealing_coef_val = _annealed_kl_weight(epoch_num, annealing_start, annealing_coef)
    loss = loss_1 + annealing_coef_val * kl_divergence(alpha, target)
    return loss


def kl_divergence(alpha, target):
    """计算 KL 散度正则化项

    用于惩罚不符合目标分布的 Dirichlet 参数，防止 evidence 无限增大。

    Args:
        alpha: Dirichlet 参数 [batch_size, num_classes]
        target: one-hot 标签 [batch_size, num_classes]

    Returns:
        KL 散度 [batch_size, 1]
    """
    num_classes = alpha.shape[1]
    # 对于目标类，alpha_t 不变；对于非目标类，alpha_nt 变为 1
    # 即只在非目标类上施加正则化
    alpha_hat = target + (1 - target) * alpha

    # Dirichlet(1,1,...,1) 是均匀分布
    beta = torch.ones_like(alpha_hat)

    S_alpha = torch.sum(alpha_hat, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    ln_B_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_hat), dim=1, keepdim=True)
    ln_B_beta = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)

    dg_alpha = torch.digamma(alpha_hat)
    dg_S_alpha = torch.digamma(S_alpha)

    kl = ln_B_alpha - ln_B_beta \
        + torch.sum((alpha_hat - beta) * (dg_alpha - dg_S_alpha), dim=1, keepdim=True)

    return kl


class EDLClassificationLoss(torch.nn.Module):
    """EDL 分类损失函数（类交叉熵版本）

    这是 EDL 的类交叉熵损失（Type II MLE），配合 KL 散度正则化使用。

    损失 = -sum_k(y_k * log(alpha_k / S)) + lambda * KL(alpha_hat || beta)
         = -sum_k(y_k * (log(alpha_k) - log(S))) + lambda * KL(alpha_hat || beta)

    其中:
        alpha_k = evidence_k + 1 (Dirichlet 参数)
        S = sum_k(alpha_k) (Dirichlet 强度)
        y_k = one-hot 标签
        lambda = 退火系数

    Args:
        num_classes: 类别数（对于二分类为2）
        evidence_type: evidence 激活函数类型 ('relu', 'exp', 'softplus')
        annealing_start: KL 退火的起始 epoch 比例
        annealing_coef: KL 退火系数的基础值
        loss_type: 'log' (类交叉熵), 'digamma', 'mse'
    """

    def __init__(self, num_classes=2, evidence_type="softplus",
                 annealing_start=0.01, annealing_coef=1.0, loss_type="log"):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_type = evidence_type
        self.annealing_start = annealing_start
        self.annealing_coef = annealing_coef
        self.loss_type = loss_type

    def forward(self, logits, targets_onehot):
        """计算 EDL 损失

        Args:
            logits: 模型原始输出 [batch_size, num_classes]
            targets_onehot: one-hot 编码标签 [batch_size, num_classes]

        Returns:
            loss: 标量损失值
        """
        evidence = get_evidence(logits, self.evidence_type)
        alpha = evidence + 1.0

        # 计算当前 epoch 数（通过 annealing_start 推算）
        # 这里我们直接使用 annealing_start 作为总 epoch 的比例
        if self.loss_type == "log":
            loss_per_sample = edl_log_loss(
                alpha,
                targets_onehot,
                epoch_num=1,
                annealing_start=self.annealing_start,
                annealing_coef=self.annealing_coef,
            )
        elif self.loss_type == "digamma":
            loss_per_sample = edl_digamma_loss(
                alpha,
                targets_onehot,
                epoch_num=1,
                annealing_start=self.annealing_start,
                annealing_coef=self.annealing_coef,
            )
        elif self.loss_type == "mse":
            loss_per_sample = edl_mse_loss(
                alpha,
                targets_onehot,
                epoch_num=1,
                annealing_start=self.annealing_start,
                annealing_coef=self.annealing_coef,
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss_per_sample.mean()


class EDLLogLossWithAnnealing(torch.nn.Module):
    """带 epoch 退火的 EDL 类交叉熵损失

    在训练过程中逐步增大 KL 散度的权重，让模型先学会分类再学会估计不确定性。

    Args:
        num_classes: 类别数
        evidence_type: evidence 激活函数类型
        total_epochs: 总训练 epoch 数（用于计算退火比例）
        annealing_start_frac: 旧版参数，表示延迟开始 KL 退火的 epoch 比例（0~1）
        annealing_coef: KL 散度的最终权重
        annealing_step: 标准 EDL 退火步长；若设置，则使用
            lambda_kl = annealing_coef * min(1, (epoch + 1) / annealing_step)
        loss_type: 'log', 'digamma', 'mse'
        class_weights: 可选类别权重，只作用于 data loss，不作用于 KL
    """

    def __init__(self, num_classes=2, evidence_type="softplus",
                 total_epochs=10, annealing_start_frac=0.0,
                 annealing_coef=1.0, annealing_step=None, loss_type="log",
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_type = evidence_type
        self.total_epochs = total_epochs
        self.annealing_start_frac = annealing_start_frac
        self.annealing_coef = annealing_coef
        self.annealing_step = None if annealing_step in (None, "") else float(annealing_step)
        self.loss_type = loss_type
        self.current_epoch = 0
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights.numel() != num_classes:
                raise ValueError(
                    f"class_weights length {class_weights.numel()} does not match num_classes={num_classes}"
                )
        self.class_weights = class_weights

    def set_epoch(self, epoch):
        """设置当前 epoch（用于退火计算）"""
        self.current_epoch = epoch

    def forward(self, logits, targets_onehot):
        """计算 EDL 损失

        Args:
            logits: 模型原始输出 [batch_size, num_classes]
            targets_onehot: one-hot 编码标签 [batch_size, num_classes]

        Returns:
            loss: 标量损失值
        """
        evidence = get_evidence(logits, self.evidence_type)
        alpha = evidence + 1.0

        # Prefer the standard EDL schedule when annealing_step is provided:
        # lambda_kl = kl_weight * min(1, (epoch + 1) / annealing_step).
        if self.annealing_step is not None:
            annealing_val = min(
                1.0,
                (float(self.current_epoch) + 1.0) / max(self.annealing_step, 1.0),
            )
        else:
            # Legacy schedule: delay KL until annealing_start_frac * total_epochs,
            # then linearly increase to 1 by the end of training.
            annealing_start_epoch = self.annealing_start_frac * self.total_epochs
            if self.current_epoch < annealing_start_epoch:
                annealing_val = 0.0
            else:
                progress = (self.current_epoch - annealing_start_epoch) / max(
                    self.total_epochs - annealing_start_epoch, 1.0
                )
                annealing_val = min(1.0, progress)

        lambda_kl = self.annealing_coef * annealing_val

        S = torch.sum(alpha, dim=1, keepdim=True)

        if self.loss_type == "log":
            # 类交叉熵: -sum_k(y_k * (log(alpha_k) - log(S)))
            loss_1 = torch.sum(
                targets_onehot * (torch.log(S + 1e-10) - torch.log(alpha + 1e-10)),
                dim=1, keepdim=True
            )
        elif self.loss_type == "digamma":
            # Digamma 版本
            loss_1 = torch.sum(
                targets_onehot * (torch.digamma(S) - torch.digamma(alpha)),
                dim=1, keepdim=True
            )
        elif self.loss_type == "mse":
            # MSE 版本
            p = alpha / S
            loss_err = torch.sum((targets_onehot - p) ** 2, dim=1, keepdim=True)
            loss_var = torch.sum(p * (1 - p) / (S + 1), dim=1, keepdim=True)
            loss_1 = loss_err + loss_var
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if self.class_weights is not None:
            target_indices = torch.argmax(targets_onehot, dim=1).long()
            sample_weights = self.class_weights.to(
                device=loss_1.device,
                dtype=loss_1.dtype,
            )[target_indices].view(-1, 1)
            loss_1 = loss_1 * sample_weights

        # KL 散度正则化项。kl_divergence 内部使用 target + (1-target) * alpha，
        # 只正则非目标类别；class_weights 不作用到这里。
        kl = kl_divergence(alpha, targets_onehot)

        loss = loss_1 + lambda_kl * kl

        return loss.mean()
