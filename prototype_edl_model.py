"""
Prototype + EDL model built on top of the Mammo-FM image encoder.

This module keeps the existing Mammo-FM backbone loading path, exposes the
sample-level embedding z, and replaces the simple classifier head with a
class-wise prototype evidential head.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "codebase"))


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


class PrototypeEDLHead(nn.Module):
    """Prototype head that turns embedding distances into class evidence."""

    def __init__(
        self,
        in_features,
        prototypes_per_class=4,
        evidence_type="softplus",
        temperature=1.0,
        similarity="neg_sq_exp",
        normalize_embeddings=True,
    ):
        super().__init__()
        if prototypes_per_class <= 0:
            raise ValueError(f"prototypes_per_class must be > 0, got {prototypes_per_class}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.in_features = int(in_features)
        self.num_classes = 2
        self.prototypes_per_class = int(prototypes_per_class)
        self.evidence_type = str(evidence_type).lower()
        self.temperature = float(temperature)
        self.similarity = str(similarity).lower()
        self.normalize_embeddings = bool(normalize_embeddings)

        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.prototypes_per_class, self.in_features) * 0.02
        )

    def _normalize(self, x):
        if not self.normalize_embeddings:
            return x
        return F.normalize(x, p=2, dim=-1, eps=1e-12)

    def initialize_prototypes(self, prototype_tensor):
        expected_shape = (self.num_classes, self.prototypes_per_class, self.in_features)
        if tuple(prototype_tensor.shape) != expected_shape:
            raise ValueError(
                f"prototype_tensor shape {tuple(prototype_tensor.shape)} does not match {expected_shape}"
            )
        with torch.no_grad():
            self.prototypes.copy_(
                prototype_tensor.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
            )

    def forward(self, z):
        if z.dim() != 2 or z.size(1) != self.in_features:
            raise ValueError(
                f"Expected z with shape [B, {self.in_features}], got {tuple(z.shape)}"
            )

        z_used = self._normalize(z)
        prototypes_used = self._normalize(self.prototypes)

        diff = z_used[:, None, None, :] - prototypes_used[None, :, :, :]
        prototype_distances = torch.sum(diff * diff, dim=-1)

        if self.similarity != "neg_sq_exp":
            raise ValueError(f"Unsupported similarity mode: {self.similarity}")
        prototype_similarities = torch.exp(-prototype_distances / self.temperature)

        # Similarities are already non-negative and directly act as prototype evidence.
        prototype_evidence = prototype_similarities
        evidence = torch.sum(prototype_evidence, dim=-1)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probability = alpha / S
        uncertainty = float(self.num_classes) / S

        return {
            "z": z,
            "prototype_distances": prototype_distances,
            "prototype_similarities": prototype_similarities,
            "prototype_evidence": prototype_evidence,
            "evidence": evidence,
            "alpha": alpha,
            "S": S,
            "probability": probability,
            "uncertainty": uncertainty,
        }


class MammoPrototypeEDLModel(nn.Module):
    """Mammo-FM image encoder plus prototype evidential head."""

    def __init__(
        self,
        args,
        ckpt,
        num_classes=2,
        evidence_type="softplus",
        prototypes_per_class=4,
        temperature=1.0,
        similarity="neg_sq_exp",
        normalize_embeddings=True,
    ):
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError(f"Prototype EDL currently supports binary classification only, got {num_classes}")

        from breastclip.model.modules import load_image_encoder

        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])

        image_encoder_weights = {}
        for key, value in ckpt["model"].items():
            if key.startswith("image_encoder."):
                image_encoder_weights[".".join(key.split(".")[1:])] = value
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        self.arch = str(getattr(args, "arch", "")).lower()
        self.freeze_backbone = _as_bool(getattr(args, "freeze_backbone", False), default=False)
        if self.freeze_backbone:
            print("Freezing image encoder; training only the Prototype EDL head")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.num_classes = 2
        self.store_features = bool(getattr(args, "store_features", False))
        self.raw_features = None
        self.pool_features = None
        self.proto_head = PrototypeEDLHead(
            in_features=self.image_encoder.out_dim,
            prototypes_per_class=prototypes_per_class,
            evidence_type=evidence_type,
            temperature=temperature,
            similarity=similarity,
            normalize_embeddings=normalize_embeddings,
        )

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):
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

        image_features = self.image_encoder(image)
        return image_features[:, 0]

    def initialize_prototypes(self, prototype_tensor):
        self.proto_head.initialize_prototypes(prototype_tensor)

    def forward_from_embedding(self, z):
        return self.proto_head(z)

    def forward(self, images):
        z = self.encode_image(images)
        return self.forward_from_embedding(z)

    def predict(self, images):
        return self.forward(images)
