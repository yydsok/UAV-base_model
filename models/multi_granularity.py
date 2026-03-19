"""
Multi-granularity feature extraction module.
Extracts token-level, object-level, and image-level features from ViT output.

Reference: SkySense++ Fine-Grained Contrastive Learning (FGCL)
- Token-level: individual patch tokens
- Object-level: Sinkhorn-Knopp clustering of patch tokens
- Image-level: global average pooling of patch tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm for soft uniform clustering with EMA centers.

    Centers are persistent (register_buffer) and updated via EMA after each
    forward pass, ensuring stable cluster assignments across iterations.
    Log-domain computation for FP16 numerical safety.
    """
    def __init__(self, num_clusters=8, num_iters=3, temperature=0.1,
                 center_ema=0.99):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.temperature = temperature
        self.center_ema = center_ema
        # Lazy-initialized on first forward (depends on embed_dim)
        self.register_buffer("centers", None)

    @torch.no_grad()
    def forward(self, features, update_centers=True):
        """
        Compute soft cluster assignments using Sinkhorn-Knopp.

        Args:
            features: [B, N, d] patch token features

        Returns:
            assignments: [B, N, K] soft assignment matrix
        """
        B, N, d = features.shape
        features_norm = F.normalize(features, dim=-1)

        # Lazy init: pick K random patches from first batch as initial centers
        if self.centers is None:
            flat = features_norm.reshape(-1, d)  # [B*N, d]
            if flat.shape[0] < self.num_clusters:
                repeat = (self.num_clusters + flat.shape[0] - 1) // flat.shape[0]
                flat = flat.repeat(repeat, 1)
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    idx = torch.randperm(flat.shape[0], device=flat.device)[:self.num_clusters]
                    centers = flat[idx].clone()
                else:
                    centers = torch.zeros(
                        self.num_clusters, d, device=flat.device, dtype=flat.dtype
                    )
                dist.broadcast(centers, src=0)
                self.centers = centers
            else:
                idx = torch.randperm(flat.shape[0], device=flat.device)[:self.num_clusters]
                self.centers = flat[idx].clone()  # [K, d]

        centers = F.normalize(self.centers, dim=-1)  # [K, d]

        logits = torch.einsum('bnd,kd->bnk', features_norm, centers)
        logits = logits / self.temperature

        Q = torch.exp(logits.float().reshape(B * N, self.num_clusters).t())  # [K, M_local]
        K, M_local = Q.shape
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        M_total = M_local * world_size

        sum_Q = Q.sum()
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q = Q / sum_Q.clamp(min=1e-12)

        for _ in range(self.num_iters):
            row_sums = Q.sum(dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(row_sums)
            Q = Q / row_sums.clamp(min=1e-12)
            Q = Q / float(K)

            col_sums = Q.sum(dim=0, keepdim=True)
            Q = Q / col_sums.clamp(min=1e-12)
            Q = Q / float(M_total)

        Q = (Q * float(M_total)).t().reshape(B, N, self.num_clusters).to(features.dtype)

        if update_centers:
            self.update_centers(features, Q)

        return Q

    @torch.no_grad()
    def update_centers(self, features, assignments):
        self.update_centers_multi([features], [assignments])

    @torch.no_grad()
    def update_centers_multi(self, features_list, assignments_list):
        total_weighted = None
        total_mass = None

        for features, assignments in zip(features_list, assignments_list):
            B, N, d = features.shape
            features_norm = F.normalize(features, dim=-1)
            Q_flat = assignments.reshape(B * N, self.num_clusters)  # [B*N, K]
            feat_flat = features_norm.reshape(B * N, d)             # [B*N, d]
            weighted = Q_flat.t() @ feat_flat                       # [K, d]
            mass = Q_flat.sum(dim=0)                                # [K]

            total_weighted = weighted if total_weighted is None else total_weighted + weighted
            total_mass = mass if total_mass is None else total_mass + mass

        if total_weighted is None or total_mass is None:
            return

        if dist.is_initialized():
            dist.all_reduce(total_weighted)
            dist.all_reduce(total_mass)

        new_centers = total_weighted / total_mass.clamp(min=1e-6).unsqueeze(-1)
        new_centers = F.normalize(new_centers, dim=-1)

        updated = (self.center_ema * F.normalize(self.centers, dim=-1)
                   + (1 - self.center_ema) * new_centers)
        self.centers = F.normalize(updated, dim=-1)


class MultiGranularityFeatures(nn.Module):
    """
    Extract multi-granularity features from ViT backbone output.

    Provides three levels of features:
    1. Token-level: [B, N, d] - individual patch tokens
    2. Object-level: [B, K, d] - Sinkhorn-Knopp cluster centers
    3. Image-level: [B, d] - global average pooling

    Each level has its own projection head for contrastive learning.
    """
    def __init__(self, embed_dim, proj_dim=256, num_clusters=8,
                 sinkhorn_iters=3, sinkhorn_temp=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_clusters = num_clusters

        # Sinkhorn-Knopp for object-level clustering
        self.sinkhorn = SinkhornKnopp(
            num_clusters=num_clusters,
            num_iters=sinkhorn_iters,
            temperature=sinkhorn_temp)

        # Projection heads for each granularity level
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.object_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, patch_tokens, assignments=None):
        """
        Args:
            patch_tokens: [B, N, d] patch token features from ViT
                         (excluding cls token)
            assignments: optional [B, N, K] soft assignments shared across
                         branches to keep object slots aligned.

        Returns:
            dict with keys:
                'token': [B, N, proj_dim] projected token features
                'object': [B, K, proj_dim] projected object (cluster) features
                'image': [B, proj_dim] projected image features
                'assignments': [B, N, K] cluster assignments (for visualization)
        """
        B, N, d = patch_tokens.shape

        # 1. Token-level features
        token_feat = self.token_proj(patch_tokens)  # [B, N, proj_dim]

        # 2. Object-level features via Sinkhorn-Knopp clustering
        if assignments is None:
            assignments = self.sinkhorn(patch_tokens)  # [B, N, K]
        # Weighted average of patch tokens per cluster
        object_feat = torch.bmm(
            assignments.transpose(1, 2),  # [B, K, N]
            patch_tokens  # [B, N, d]
        )  # [B, K, d]
        # Normalize by cluster size
        cluster_sizes = assignments.sum(dim=1, keepdim=False).clamp(min=1e-6)  # [B, K]
        object_feat = object_feat / cluster_sizes.unsqueeze(-1)
        object_feat = self.object_proj(object_feat)  # [B, K, proj_dim]

        # 3. Image-level features via global average pooling
        image_feat = patch_tokens.mean(dim=1)  # [B, d]
        image_feat = self.image_proj(image_feat)  # [B, proj_dim]

        return {
            'token': token_feat,
            'object': object_feat,
            'image': image_feat,
            'assignments': assignments,
        }
