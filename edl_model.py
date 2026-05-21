"""
Evidential Deep Learning (EDL) 模型模块

在 Mammo-FM 骨干网络基础上替换分类头为 EDL 头，
输出 evidence 用于构造 Dirichlet 分布，从而进行不确定性估计。

不修改原项目代码，完全独立模块。
"""

import torch
import torch.nn as nn

from breastclip.model.modules import load_image_encoder
from edl_loss import get_evidence


class EDLClassifier(nn.Module):
    """EDL 分类头

    将特征映射到 num_classes 维 logits，再通过可配置的 evidence 激活函数
    构造 Dirichlet 参数 alpha = evidence + 1。

    Args:
        feature_dim: 输入特征维度
        num_classes: 类别数（二分类为2）
    """

    def __init__(self, feature_dim, num_classes=2):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        """输出原始 logits，后续通过 evidence 激活函数转换"""
        return self.classification_head(x)


class MammoEDLModel(nn.Module):
    """基于 Mammo-FM 骨干网络的 EDL 模型

    加载 Mammo-FM 预训练的图像编码器，替换最后的线性分类头为 EDL 分类头。
    骨干网络是否冻结由 args.freeze_backbone 显式控制，不再依赖 arch 名称。

    模型输出:
        - logits: 原始输出 [B, num_classes]
        - 经过 evidence 激活后可得:
            - evidence = get_evidence(logits)  [B, num_classes]
            - alpha = evidence + 1     [B, num_classes]
            - probability = alpha / sum(alpha)  [B, num_classes]
            - uncertainty = num_classes / sum(alpha)  [B, 1]

    Args:
        args: 包含 arch 等配置的 Namespace 对象
        ckpt: Mammo-FM 预训练 checkpoint (dict)
        num_classes: 类别数（默认2，即二分类）
        evidence_type: evidence 激活函数类型 ('relu', 'exp', 'softplus')
    """

    def __init__(self, args, ckpt, num_classes=2, evidence_type="softplus"):
        super().__init__()

        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])

        # 加载预训练权重
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        self.arch = str(getattr(args, "arch", "")).lower()
        freeze_backbone = getattr(args, "freeze_backbone", False)
        if isinstance(freeze_backbone, str):
            freeze_backbone = freeze_backbone.lower() in {"1", "true", "t", "yes", "y"}
        self.freeze_backbone = bool(freeze_backbone)
        self.evidence_type = evidence_type
        self.num_classes = num_classes

        if self.freeze_backbone:
            print("Freezing image encoder; training only the EDL head")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # EDL 分类头：输出 num_classes 维 evidence
        self.classifier = EDLClassifier(
            feature_dim=self.image_encoder.out_dim,
            num_classes=num_classes
        )

        self.store_features = bool(getattr(args, "store_features", False))
        self.raw_features = None
        self.pool_features = None

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):
        """提取图像特征"""
        if self.image_encoder_type == "cnn":
            input_dict = {"image": image, "breast_clip_train_mode": True}
            image_features, raw_features = self.image_encoder(input_dict)
            if self.store_features:
                self.raw_features = raw_features.detach()
                self.pool_features = image_features.detach()
            else:
                self.raw_features = None
                self.pool_features = None
            return image_features
        else:
            image_features = self.image_encoder(image)
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def forward(self, images):
        """前向传播

        Args:
            images: 输入图像 [B, C, H, W]

        Returns:
            logits: 原始输出 [B, num_classes]
        """
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits

    def predict(self, images):
        """预测接口，返回完整的 EDL 分析结果

        Args:
            images: 输入图像 [B, C, H, W]

        Returns:
            dict containing:
                - logits: 原始输出 [B, num_classes]
                - evidence: evidence 值 [B, num_classes]
                - alpha: Dirichlet 参数 [B, num_classes]
                - probability: 预测概率 [B, num_classes]
                - uncertainty: 不确定性 [B, 1]
        """
        logits = self.forward(images)
        evidence = self._get_evidence(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probability = alpha / S
        uncertainty = self.num_classes / S

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "probability": probability,
            "uncertainty": uncertainty,
        }

    def _get_evidence(self, logits):
        """将 logits 转换为 evidence"""
        return get_evidence(logits, self.evidence_type)
