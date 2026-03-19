"""
DINO projection head and MultiCropWrapper.
Based on the original DINO implementation.
"""

import torch
import torch.nn as nn


class DINOHead(nn.Module):
    """
    DINO projection head: MLP with L2 normalization.
    Projects backbone features to a high-dimensional space for contrastive learning.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Wraps backbone + head to handle multi-crop inputs efficiently.
    Groups crops by resolution and processes them in batches.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, return_backbone_feat=False, modality_masks=None):
        # x is a list of tensors with potentially different sizes
        # Group crops by resolution for efficient batching
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        output_cls = torch.empty(0).to(x[0].device)
        output_tokens_by_group = []  # Keep separate per resolution group

        for end_idx in idx_crops:
            # Batch crops of the same resolution
            inp = torch.cat(x[start_idx:end_idx])
            n_crops_in_group = end_idx - start_idx

            # Repeat modality_masks for each crop in the group
            group_masks = None
            if modality_masks is not None:
                group_masks = modality_masks.repeat(n_crops_in_group, 1)

            # Get all tokens (cls + patches)
            _out = self.backbone(
                inp, return_all_tokens=True, modality_masks=group_masks)

            # Separate cls token and patch tokens
            cls_token = _out[:, 0]  # [B, d]
            patch_tokens = _out[:, 1:]  # [B, N, d]

            # Project cls token through head
            cls_proj = self.head(cls_token)
            output_cls = torch.cat((output_cls, cls_proj))
            output_tokens_by_group.append(patch_tokens)

            start_idx = end_idx

        if return_backbone_feat:
            # Only return tokens from the first resolution group (global crops)
            # because different resolutions have different token counts
            # and the training loop only uses global crop tokens anyway.
            return output_cls, output_tokens_by_group[0]
        return output_cls
