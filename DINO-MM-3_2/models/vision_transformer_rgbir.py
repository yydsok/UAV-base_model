"""
Vision Transformer backbone adapted for 4-channel (RGB+IR) input.
Based on DINO-MM's ViT implementation with modifications for:
- 4-channel input (3 RGB + 1 IR)
- Multi-granularity feature output (token/object/image level)
- Optional temporal attention slot for video frames
"""

import math
import torch
import torch.nn as nn
from functools import partial


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding, supports arbitrary input channels."""
    def __init__(self, img_size=224, patch_size=16, in_chans=4, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # Conv2d projects all input channels (RGB+IR=4) to embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] where C=4 (RGB+IR)
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class DualModalPatchEmbed(nn.Module):
    """
    Separate patch embeddings for RGB (3ch) and IR (1ch), inspired by SkySense++.

    Instead of concatenating RGB+IR into a 4-channel input before a single Conv,
    each modality gets its own dedicated patch projection. This allows:
      - Direct loading of 3-channel pretrained weights into rgb_proj (no channel hacks)
      - Modality-specific feature extraction before any cross-modal interaction
      - Flexible fusion strategies

    Fusion modes:
      'concat'     : cat([rgb, ir], dim=-1) → Linear(2d→d) + LayerNorm  [recommended]
      'add'        : rgb_tokens + ir_tokens  (symmetric, no extra params)
      'cross_attn' : RGB tokens cross-attend to IR tokens (RGB enhanced by IR context)
    """
    def __init__(self, img_size=224, patch_size=16,
                 rgb_chans=3, ir_chans=1, embed_dim=768,
                 fusion='concat'):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.fusion = fusion

        # Per-modality patch projections (each learns its own feature space)
        self.rgb_proj = nn.Conv2d(
            rgb_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.ir_proj = nn.Conv2d(
            ir_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if fusion == 'concat':
            # Project concatenated features back to embed_dim
            self.fusion_proj = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        elif fusion == 'cross_attn':
            # RGB queries attend to IR context; residual preserves RGB features
            num_heads = max(1, embed_dim // 64)
            self.rgb_norm = nn.LayerNorm(embed_dim)
            self.ir_norm = nn.LayerNorm(embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, num_heads=num_heads, batch_first=True)
            self.fusion_norm = nn.LayerNorm(embed_dim)
        # 'add': no extra parameters needed

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W]  channels 0:3=RGB, channel 3:=IR
               When a modality is dropped (RandomSensorDrop), those channels are 0.
        Returns:
            fused: [B, N, embed_dim]
        """
        rgb = x[:, :3]   # [B, 3, H, W]
        ir  = x[:, 3:]   # [B, 1, H, W]

        # Per-modality patch tokens
        rgb_tok = self.rgb_proj(rgb).flatten(2).transpose(1, 2)  # [B, N, d]
        ir_tok  = self.ir_proj(ir).flatten(2).transpose(1, 2)    # [B, N, d]

        if self.fusion == 'concat':
            return self.fusion_proj(torch.cat([rgb_tok, ir_tok], dim=-1))
        elif self.fusion == 'add':
            return rgb_tok + ir_tok
        elif self.fusion == 'cross_attn':
            # RGB queries, IR keys/values; residual connection keeps RGB grounded
            q = self.rgb_norm(rgb_tok)
            k = v = self.ir_norm(ir_tok)
            attn_out, _ = self.cross_attn(q, k, v)
            return self.fusion_norm(rgb_tok + attn_out)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion!r}")


class VisionTransformer(nn.Module):
    """
    ViT backbone for multi-modal (RGB+IR) DINO pretraining.

    Outputs multi-granularity features:
    - cls_token: [B, d] for image-level representation
    - patch_tokens: [B, N, d] for token-level representation
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=4, num_classes=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, fusion='concat', **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # Use DualModalPatchEmbed for 4-channel (RGB+IR) input so each modality
        # has its own patch projection before fusion, following SkySense++ design.
        if in_chans == 4:
            self.patch_embed = DualModalPatchEmbed(
                img_size=img_size, patch_size=patch_size,
                rgb_chans=3, ir_chans=1, embed_dim=embed_dim,
                fusion=fusion)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size,
                in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classification head (unused in pretraining, kept for compatibility)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate position embeddings for different input sizes."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # Add small number to avoid interpolation artifacts
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # [B, N, d]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=False):
        """
        Args:
            x: [B, 4, H, W] input tensor (RGB 3ch + IR 1ch)
            return_all_tokens: if True, return all patch tokens

        Returns:
            If return_all_tokens=True: [B, N+1, d] (cls + patch tokens)
            Else: [B, d] (cls token only)
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_tokens:
            return x  # [B, N+1, d] -- cls_token at index 0, patch tokens at 1:
        return x[:, 0]  # [B, d] -- cls token only

    def get_intermediate_layers(self, x, n=1):
        """Get outputs from the last n transformer blocks."""
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


# ============================================================================
# Factory functions for different ViT sizes
# ============================================================================

def vit_tiny(patch_size=16, fusion='concat', **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fusion=fusion, **kwargs)
    return model


def vit_small(patch_size=16, fusion='concat', **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fusion=fusion, **kwargs)
    return model


def vit_base(patch_size=16, fusion='concat', **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fusion=fusion, **kwargs)
    return model


def vit_large(patch_size=16, fusion='concat', **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fusion=fusion, **kwargs)
    return model


def vit_huge(patch_size=14, fusion='concat', **kwargs):
    """ViT-H/14: ~632M backbone parameters."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fusion=fusion, **kwargs)
    return model


def vit_giant(patch_size=14, fusion='concat', **kwargs):
    """ViT-g/14: ~1.1B backbone parameters. Target architecture for 1B base model."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16,
        mlp_ratio=48 / 11,  # matches DINOv2 ViT-g ffn_dim=6144 (1408*48/11≈6144)
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fusion=fusion, **kwargs)
    return model
