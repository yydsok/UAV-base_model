"""
View-Domain Bridge module.

Projects image-level features to a shared embedding space with learnable
prototypes. The prototype logits are used for view bridge loss to improve
cross-view and cross-dataset semantic consistency without view labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewDomainBridge(nn.Module):
    """Project features and produce prototype logits for bridge loss."""

    def __init__(self, in_dim, proj_dim=256, num_prototypes=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proj_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, in_dim] image-level features

        Returns:
            z: [B, proj_dim] normalized projected features
            logits: [B, K] similarity logits to prototypes
        """
        z = self.projector(x)
        z = F.normalize(z, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = z @ proto.t()
        return z, logits
