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


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm for soft uniform clustering.
    Used to group patch tokens into object-level clusters.
    """
    def __init__(self, num_clusters=8, num_iters=3, temperature=0.1):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, features):
        """
        Compute soft cluster assignments using Sinkhorn-Knopp.

        Args:
            features: [B, N, d] patch token features

        Returns:
            assignments: [B, N, K] soft assignment matrix
        """
        B, N, d = features.shape
        # Normalize features
        features_norm = F.normalize(features, dim=-1)

        # Initialize cluster centers as random feature subsets
        indices = torch.randint(0, N, (self.num_clusters,), device=features.device)
        centers = features_norm[:, indices, :]  # [B, K, d]

        # Compute assignment logits
        logits = torch.bmm(features_norm, centers.transpose(1, 2))  # [B, N, K]
        logits = logits / self.temperature

        # Sinkhorn-Knopp iterations for uniform assignment
        Q = torch.exp(logits)
        for _ in range(self.num_iters):
            # Row normalization
            Q = Q / Q.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            # Column normalization (enforce uniform cluster sizes)
            Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Final normalization
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return Q


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

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, N, d] patch token features from ViT
                         (excluding cls token)

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
