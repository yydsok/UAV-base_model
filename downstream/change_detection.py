import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import build_backbone_from_checkpoint, freeze_backbone
from downstream.models import DinoMMDenseBackbone


class DinoMMBITChangeDetector(nn.Module):
    """BIT-style change detector with a DINO-MM dense backbone.

    Improvements over vanilla BIT:
      - Multi-scale feature fusion (all 4 FPN levels fused to hidden_dim)
      - Deep supervision via auxiliary classifier on raw diff features
    """

    def __init__(
        self,
        dense_backbone,
        hidden_dim=128,
        token_len=4,
        encoder_layers=2,
        decoder_layers=1,
        num_heads=4,
    ):
        super().__init__()
        self.backbone = dense_backbone
        self.hidden_dim = hidden_dim
        self.token_len = token_len
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # Multi-scale fusion: project each FPN level to hidden_dim, then fuse
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dense_backbone.out_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.token_proj = nn.Conv2d(hidden_dim, token_len, kernel_size=1, bias=False)
        self.token_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=encoder_layers,
        )
        self.map_decoder = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )
        # Auxiliary deep supervision head on raw diff (before token exchange)
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1),
        )

    def _tokenize(self, feature_map):
        batch_size, channels, height, width = feature_map.shape
        attn_logits = self.token_proj(feature_map).view(batch_size, self.token_len, -1)
        attn = torch.softmax(attn_logits, dim=-1)
        flattened = feature_map.view(batch_size, channels, -1)
        tokens = torch.einsum("btn,bcn->btc", attn, flattened)
        return tokens

    def _decode(self, feature_map, tokens):
        batch_size, channels, height, width = feature_map.shape
        flattened = feature_map.flatten(2).transpose(1, 2)
        for attn in self.map_decoder:
            decoded, _ = attn(flattened, tokens, tokens, need_weights=False)
            flattened = flattened + decoded
        return flattened.transpose(1, 2).reshape(batch_size, channels, height, width)

    def _extract_map(self, images):
        features = self.backbone.forward_feature_list(images)
        # Fuse all 4 FPN levels to a common spatial resolution (features[1] size)
        target_size = features[1].shape[-2:]
        projected = []
        for feat, lateral in zip(features, self.lateral_convs):
            proj = lateral(feat)
            if proj.shape[-2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj)
        return self.fuse_conv(torch.cat(projected, dim=1))

    def forward(self, image_a, image_b):
        map_a = self._extract_map(image_a)
        map_b = self._extract_map(image_b)

        # Auxiliary output on raw diff (deep supervision, before token exchange)
        raw_diff = torch.abs(map_a - map_b)
        aux_logits = self.aux_classifier(raw_diff)
        aux_logits = F.interpolate(aux_logits, size=image_a.shape[-2:], mode="bilinear", align_corners=False)

        token_a = self._tokenize(map_a)
        token_b = self._tokenize(map_b)
        fused_tokens = self.token_encoder(torch.cat([token_a, token_b], dim=1))
        token_a, token_b = fused_tokens.chunk(2, dim=1)

        map_a = self._decode(map_a, token_a)
        map_b = self._decode(map_b, token_b)

        diff_map = torch.abs(map_a - map_b)
        logits = self.classifier(diff_map)
        logits = F.interpolate(logits, size=image_a.shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            return logits, aux_logits
        return logits


def build_change_detection_model(
    checkpoint_path,
    checkpoint_key="teacher",
    device="cpu",
    init_mode="pretrained",
    arch=None,
    patch_size=None,
    in_chans=None,
    fusion=None,
    trainable_blocks=4,
    feature_dim=256,
    hidden_dim=128,
    token_len=4,
    encoder_layers=2,
    decoder_layers=1,
    num_heads=4,
):
    backbone, _, meta = build_backbone_from_checkpoint(
        checkpoint_path,
        checkpoint_key=checkpoint_key,
        device=device,
        arch=arch,
        patch_size=patch_size,
        in_chans=in_chans,
        fusion=fusion,
        load_temporal=False,
        init_mode=init_mode,
    )
    freeze_backbone(backbone, trainable_blocks=trainable_blocks, train_patch_embed=False)
    dense_backbone = DinoMMDenseBackbone(backbone, feature_dim=feature_dim)
    dense_backbone.to(device)

    model = DinoMMBITChangeDetector(
        dense_backbone=dense_backbone,
        hidden_dim=hidden_dim,
        token_len=token_len,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
    )
    model.to(device)
    meta["framework"] = "bit_dino_mm"
    meta["feature_dim"] = feature_dim
    meta["hidden_dim"] = hidden_dim
    meta["token_len"] = token_len
    return model, meta
