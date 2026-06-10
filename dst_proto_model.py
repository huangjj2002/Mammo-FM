import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


def dst_activation_init_gamma(gamma_init):
    return math.sqrt(max(float(gamma_init), 1e-6))


def dst_activation_init_alpha(alpha_init):
    return float(alpha_init)


def pignistic(mass, n_class):
    class_mass = mass[..., :n_class]
    omega = mass[..., n_class]
    probs = class_mass + (1.0 / n_class) * omega.unsqueeze(-1)
    return probs, omega


class PrototypeDSTNLLLoss(nn.Module):
    """NLL on pignistic probabilities, matching the current DST prototype script."""

    def __init__(self, weight=None, eps=1e-10):
        super().__init__()
        self.nll_loss = nn.NLLLoss(weight=weight)
        self.eps = eps

    def forward(self, head_output, targets):
        probs = head_output["prob"].clamp(min=self.eps)
        weight = self.nll_loss.weight
        if weight is not None:
            weight = weight.to(device=probs.device, dtype=probs.dtype)
        return F.nll_loss(torch.log(probs), targets.long(), weight=weight)


class DSTPrototypeHead(nn.Module):
    """Class-wise prototype head with Dempster-Shafer mass combination."""

    def __init__(
        self,
        in_features,
        num_classes=2,
        prototypes_per_class=4,
        topk=3,
        normalize=True,
        gamma_init=1.0,
        alpha_init=0.0,
        dropout=0.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("DSTPrototypeHead currently expects binary output.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be >= 1.")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.n_prototypes = self.num_classes * self.prototypes_per_class
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.normalize_embeddings = self.normalize
        self.gamma_init = float(gamma_init)
        self.alpha_init = float(alpha_init)
        self.drop = nn.Dropout(p=float(dropout))

        init_gamma = dst_activation_init_gamma(self.gamma_init)
        init_alpha = dst_activation_init_alpha(self.alpha_init)
        self.prototype_weight = nn.Parameter(
            torch.empty(self.n_prototypes, self.in_features)
        )
        self.activation_alpha = nn.Parameter(torch.full((self.n_prototypes,), init_alpha))
        self.activation_eta = nn.Parameter(torch.full((self.n_prototypes,), init_gamma))
        self.reset_parameters()

    @property
    def prototypes(self):
        return self.prototype_weight.view(
            self.num_classes, self.prototypes_per_class, self.in_features
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prototype_weight)

    def _compute_distances(self, x):
        prototypes = self.prototype_weight
        if self.normalize:
            x = F.normalize(x, dim=-1, eps=1e-12)
            prototypes = F.normalize(prototypes, dim=-1, eps=1e-12)
        return (x[:, None, :] - prototypes[None, :, :]).pow(2).sum(dim=-1)

    def _distance_activation(self, distances):
        alpha = torch.sigmoid(self.activation_alpha).view(1, -1)
        eta_sq = self.activation_eta.pow(2).view(1, -1)
        return alpha * torch.exp(-eta_sq * distances)

    def _prototype_masses(self, similarities):
        batch_size = similarities.shape[0]
        mass = similarities.new_zeros(
            batch_size, self.n_prototypes, self.num_classes + 1
        )
        omega_idx = self.num_classes
        proto_classes = torch.arange(
            self.n_prototypes, device=similarities.device
        ) // self.prototypes_per_class
        mass.scatter_(2, proto_classes.view(1, -1, 1).expand(batch_size, -1, 1), similarities.unsqueeze(-1))
        mass[..., omega_idx] = (1.0 - similarities).clamp(min=1e-8, max=1.0)
        return mass

    def _combine_pair(self, current, incoming):
        current_class = current[:, : self.num_classes]
        incoming_class = incoming[:, : self.num_classes]
        current_omega = current[:, self.num_classes : self.num_classes + 1]
        incoming_omega = incoming[:, self.num_classes : self.num_classes + 1]

        same_class = current_class * incoming_class
        incoming_unknown = current_class * incoming_omega
        current_unknown = incoming_class * current_omega
        class_mass = same_class + incoming_unknown + current_unknown
        omega = current_omega * incoming_omega

        numerator = torch.cat([class_mass, omega], dim=-1).clamp_min(0.0)
        normalizer = numerator.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return numerator / normalizer

    def _combine_masses(self, mass_prototypes):
        combined = mass_prototypes[:, 0, :]
        for proto_idx in range(1, mass_prototypes.size(1)):
            combined = self._combine_pair(combined, mass_prototypes[:, proto_idx, :])
        return combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _reshape_prototypes(self, tensor):
        batch_size = tensor.shape[0]
        return tensor.view(batch_size, self.num_classes, self.prototypes_per_class)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Expected x with shape [B, {self.in_features}], got {tuple(x.shape)}")

        x = self.drop(x)
        distances = self._compute_distances(x)
        similarity_flat = self._distance_activation(distances)
        mass_prototypes = self._prototype_masses(similarity_flat)
        mass = self._combine_masses(mass_prototypes)
        prob, uncertainty = pignistic(mass, self.num_classes)

        prototype_evidence = torch.zeros(
            x.shape[0],
            self.num_classes,
            self.prototypes_per_class,
            device=x.device,
            dtype=x.dtype,
        )
        for class_idx in range(self.num_classes):
            start = class_idx * self.prototypes_per_class
            end = start + self.prototypes_per_class
            prototype_evidence[:, class_idx, :] = mass_prototypes[:, start:end, class_idx]

        distances_by_class = self._reshape_prototypes(distances)
        similarity = self._reshape_prototypes(similarity_flat)

        out = {
            "prob": prob,
            "probability": prob,
            "uncertainty": uncertainty,
            "dst_mass": mass,
            "prototype_distances": distances_by_class,
            "prototype_similarity": similarity,
            "prototype_similarities": similarity,
            "prototype_evidence": prototype_evidence,
            "prototype_mass": prototype_evidence,
        }

        if self.topk > 0:
            topk = min(self.topk, self.prototypes_per_class)
            top_evidence, top_idx = torch.topk(prototype_evidence, k=topk, dim=-1)
            top_similarity = torch.gather(similarity, dim=-1, index=top_idx)
            top_distances = torch.gather(distances_by_class, dim=-1, index=top_idx)
            out.update(
                {
                    "topk_proto_idx": top_idx,
                    "topk_proto_evidence": top_evidence,
                    "topk_proto_mass": top_evidence,
                    "topk_proto_similarity": top_similarity,
                    "topk_proto_distances": top_distances,
                }
            )

        return out

    def initialize_from_embeddings(self, embeddings, labels, random_state=0):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().float().numpy()
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels).astype(int)

        if embeddings.ndim != 2 or embeddings.shape[1] != self.in_features:
            raise ValueError(
                f"Expected embeddings with shape (N, {self.in_features}), got {embeddings.shape}."
            )
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must contain the same number of rows.")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot initialize prototypes from an empty embedding set.")

        working_embeddings = embeddings
        if self.normalize:
            norms = np.linalg.norm(working_embeddings, axis=1, keepdims=True)
            working_embeddings = working_embeddings / np.clip(norms, 1e-12, None)

        global_center = working_embeddings.mean(axis=0, keepdims=True)
        centers_by_class = []
        warnings = []

        for class_idx in range(self.num_classes):
            class_embeddings = working_embeddings[labels == class_idx]
            if len(class_embeddings) == 0:
                centers = np.repeat(global_center, self.prototypes_per_class, axis=0)
                warnings.append(f"class {class_idx} has no samples; using global mean prototypes")
            elif len(class_embeddings) >= self.prototypes_per_class:
                kmeans = KMeans(
                    n_clusters=self.prototypes_per_class,
                    n_init=10,
                    random_state=random_state + class_idx,
                )
                centers = kmeans.fit(class_embeddings).cluster_centers_.astype(np.float32)
            else:
                centers = class_embeddings.astype(np.float32)
                repeat_idx = 0
                while len(centers) < self.prototypes_per_class:
                    centers = np.concatenate(
                        [centers, class_embeddings[[repeat_idx % len(class_embeddings)]]],
                        axis=0,
                    )
                    repeat_idx += 1
                warnings.append(
                    f"class {class_idx} has fewer samples than prototypes; repeated centers"
                )

            centers_by_class.append(centers[: self.prototypes_per_class])

        centers = np.stack(centers_by_class, axis=0).astype(np.float32)
        flat_centers = centers.reshape(self.n_prototypes, self.in_features)
        with torch.no_grad():
            self.prototype_weight.copy_(
                torch.as_tensor(
                    flat_centers,
                    dtype=self.prototype_weight.dtype,
                    device=self.prototype_weight.device,
                )
            )

        return warnings


class DSTPrototypeModel(nn.Module):
    def __init__(
        self,
        in_features,
        num_classes=2,
        prototypes_per_class=4,
        topk=3,
        normalize=True,
        gamma_init=1.0,
        alpha_init=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.proto_head = DSTPrototypeHead(
            in_features=in_features,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            topk=topk,
            normalize=normalize,
            gamma_init=gamma_init,
            alpha_init=alpha_init,
            dropout=dropout,
        )

    def forward(self, embeddings):
        return self.proto_head(embeddings)

    def initialize_from_embeddings(self, embeddings, labels, random_state=0):
        return self.proto_head.initialize_from_embeddings(
            embeddings,
            labels,
            random_state=random_state,
        )
