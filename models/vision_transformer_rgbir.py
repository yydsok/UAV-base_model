"""
Vision Transformer backbone adapted for 4-channel (RGB+IR) input.
Based on DINO-MM's ViT implementation with modifications for:
- 4-channel input (3 RGB + 1 IR)
- Two-stage architecture: Stage1 (independent per-modality) + Stage2 (fused)
- CLIP-style cross-modal alignment at Stage1 output
- Multi-granularity feature output (token/object/image level)
- Optional temporal attention slot for video frames
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
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

    def forward(self, x, modality_masks=None):
        # x: [B, C, H, W] where C=4 (RGB+IR)
        # modality_masks is accepted but unused (only for DualModalPatchEmbed compat)
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

    When a modality is missing (indicated by modality_masks), learnable missing
    tokens are substituted instead of passing zero-tensors through Conv2d
    (which produces non-zero bias output that corrupts representations).

    Fusion modes:
      'concat'           : cat([rgb, ir], dim=-1) → Linear(2d→d) + LayerNorm
      'add'              : rgb_tokens + ir_tokens  (symmetric, no extra params)
      'cross_attn'       : RGB tokens cross-attend to IR tokens (RGB enhanced by IR context)
      'gated_cross_attn' : Bidirectional gated cross-attention with sigmoid gates [recommended]
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

        # Learnable missing-modality tokens: used when a modality is absent
        # instead of passing zero-tensors through Conv2d (avoids bias leakage)
        self.missing_rgb_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.missing_ir_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.missing_rgb_token, std=.02)
        nn.init.trunc_normal_(self.missing_ir_token, std=.02)

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
        elif fusion == 'gated_cross_attn':
            self.gated_cross_attn = GatedCrossAttention(embed_dim)
        # 'add': no extra parameters needed

    def forward(self, x, modality_masks=None):
        """
        Args:
            x: [B, 4, H, W]  channels 0:3=RGB, channel 3:=IR
               When a modality is dropped (RandomSensorDrop), those channels are 0.
            modality_masks: [B, 2] binary mask.
                            [:,0]=RGB available, [:,1]=IR available.
                            None means both available for all samples.
        Returns:
            fused: [B, N, embed_dim]
        """
        B = x.shape[0]
        rgb = x[:, :3]   # [B, 3, H, W]
        ir  = x[:, 3:]   # [B, 1, H, W]

        # Determine which samples have each modality available
        if modality_masks is not None:
            rgb_avail = modality_masks[:, 0].bool()  # [B]
            ir_avail = modality_masks[:, 1].bool()    # [B]
        else:
            rgb_avail = torch.ones(B, dtype=torch.bool, device=x.device)
            ir_avail = torch.ones(B, dtype=torch.bool, device=x.device)

        # --- RGB/IR tokens ---
        # Always run per-modality projection, then replace missing modalities
        # with learnable missing tokens. This keeps dtype/device aligned with
        # autocast outputs and avoids index_put dtype mismatch.
        rgb_tok = self.rgb_proj(rgb).flatten(2).transpose(1, 2)  # [B, N, d]
        ir_tok = self.ir_proj(ir).flatten(2).transpose(1, 2)     # [B, N, d]
        N = rgb_tok.shape[1]

        if not rgb_avail.all():
            missing_rgb = self.missing_rgb_token.to(dtype=rgb_tok.dtype, device=rgb_tok.device)
            rgb_tok = torch.where(
                rgb_avail.view(B, 1, 1),
                rgb_tok,
                missing_rgb.expand(B, N, -1),
            )

        if not ir_avail.all():
            missing_ir = self.missing_ir_token.to(dtype=ir_tok.dtype, device=ir_tok.device)
            ir_tok = torch.where(
                ir_avail.view(B, 1, 1),
                ir_tok,
                missing_ir.expand(B, N, -1),
            )

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
        elif self.fusion == 'gated_cross_attn':
            return self.gated_cross_attn(rgb_tok, ir_tok, modality_masks)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion!r}")

    def forward_dual_stream(self, x, modality_masks=None):
        """Like forward() but returns (rgb_tok, ir_tok) BEFORE fusion.

        Used by the dual-stream intermediate cross-attention path.
        """
        B = x.shape[0]
        rgb = x[:, :3]
        ir = x[:, 3:]

        if modality_masks is not None:
            rgb_avail = modality_masks[:, 0].bool()
            ir_avail = modality_masks[:, 1].bool()
        else:
            rgb_avail = torch.ones(B, dtype=torch.bool, device=x.device)
            ir_avail = torch.ones(B, dtype=torch.bool, device=x.device)

        rgb_tok = self.rgb_proj(rgb).flatten(2).transpose(1, 2)
        ir_tok = self.ir_proj(ir).flatten(2).transpose(1, 2)
        N = rgb_tok.shape[1]

        if not rgb_avail.all():
            missing_rgb = self.missing_rgb_token.to(dtype=rgb_tok.dtype, device=rgb_tok.device)
            rgb_tok = torch.where(rgb_avail.view(B, 1, 1), rgb_tok, missing_rgb.expand(B, N, -1))

        if not ir_avail.all():
            missing_ir = self.missing_ir_token.to(dtype=ir_tok.dtype, device=ir_tok.device)
            ir_tok = torch.where(ir_avail.view(B, 1, 1), ir_tok, missing_ir.expand(B, N, -1))

        return rgb_tok, ir_tok


class GatedCrossAttention(nn.Module):
    """RGB <-> IR bidirectional cross-attention with sigmoid gating.

    When a modality is missing (indicated by modality_masks), its tokens are
    masked in the key_padding_mask so the cross-attention output is zero,
    and the gate prevents noise injection.
    """
    def __init__(self, embed_dim, num_heads=None):
        super().__init__()
        num_heads = num_heads or max(1, embed_dim // 64)
        self.rgb_to_ir_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.ir_to_rgb_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.rgb_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ir_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, rgb_tok, ir_tok, modality_masks=None):
        """
        Args:
            rgb_tok: [B, N, D] RGB patch tokens
            ir_tok: [B, N, D] IR patch tokens
            modality_masks: [B, 2] binary mask. [:,0]=RGB avail, [:,1]=IR avail.
                            None means both available.
        Returns:
            fused: [B, N, D]
        """
        # Run cross-attention WITHOUT key_padding_mask to avoid NaN when all
        # keys are masked for a sample. Instead, zero out the contribution
        # for missing modalities via the gate mask.
        ir_ctx, _ = self.rgb_to_ir_attn(rgb_tok, ir_tok, ir_tok)
        rgb_ctx, _ = self.ir_to_rgb_attn(ir_tok, rgb_tok, rgb_tok)

        if modality_masks is not None:
            # [B, 1, 1] masks: zero out cross-modal contribution when source is missing
            ir_avail = modality_masks[:, 1].view(-1, 1, 1)   # IR available → enhance RGB
            rgb_avail = modality_masks[:, 0].view(-1, 1, 1)  # RGB available → enhance IR
            rgb_enhanced = rgb_tok + torch.sigmoid(self.rgb_gate) * ir_ctx * ir_avail
            ir_enhanced = ir_tok + torch.sigmoid(self.ir_gate) * rgb_ctx * rgb_avail
        else:
            rgb_enhanced = rgb_tok + torch.sigmoid(self.rgb_gate) * ir_ctx
            ir_enhanced = ir_tok + torch.sigmoid(self.ir_gate) * rgb_ctx

        fused = self.fusion_proj(
            torch.cat([rgb_enhanced, ir_enhanced], dim=-1))
        return fused


class IntermediateCrossAttention(nn.Module):
    """Cross-attention module for dual-stream intermediate layers.

    Applied at specified transformer layers to exchange information between
    two streams (e.g., RGB-dominant and IR-dominant). Uses a sigmoid gate
    initialized near zero to preserve the original stream behavior early
    in training.
    """

    def __init__(self, embed_dim, num_heads=None):
        super().__init__()
        num_heads = num_heads or max(1, embed_dim // 64)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        # Initialize gate to sigmoid(-5.0) ≈ 0.007 so cross-attention
        # has near-zero contribution initially
        self.gate = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x_query, x_context):
        """x_query attends to x_context with gated residual."""
        q = self.norm_q(x_query)
        kv = self.norm_kv(x_context)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return x_query + torch.sigmoid(self.gate) * attn_out


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
                 drop_path_rate=0., norm_layer=nn.LayerNorm, fusion='concat',
                 use_gradient_checkpointing=False,
                 intermediate_cross_attn_layers=None,
                 fusion_start_block=0, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.intermediate_cross_attn_layers = intermediate_cross_attn_layers or []
        if fusion_start_block < 0 or fusion_start_block >= depth:
            raise ValueError(
                f"fusion_start_block={fusion_start_block} must be in [0, {depth-1}]")
        self.fusion_start_block = fusion_start_block

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

        # Two-stage split: Stage1 (independent per-modality) + Stage2 (fused)
        # When fusion_start_block > 0 and patch_embed is DualModal:
        #   Stage1 = blocks[0:fusion_start_block] — RGB and IR go through independently
        #   Stage2 = blocks[fusion_start_block:] — fused tokens go through together
        #   GatedCrossAttention sits between Stage1 and Stage2
        self._use_two_stage = (fusion_start_block > 0
                               and isinstance(self.patch_embed, DualModalPatchEmbed))
        if self._use_two_stage:
            # Fusion layer between Stage1 and Stage2
            self.mid_fusion = GatedCrossAttention(embed_dim)
            # Separate CLS tokens for RGB and IR streams in Stage1
            self.cls_token_rgb = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_token_ir = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token_rgb, std=.02)
            nn.init.trunc_normal_(self.cls_token_ir, std=.02)
            # Cache for Stage1 outputs (read by external CLIP projector)
            self._stage1_rgb = None
            self._stage1_ir = None

        # Classification head (unused in pretraining, kept for compatibility)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # Intermediate cross-attention for dual-stream path
        if self.intermediate_cross_attn_layers:
            self.intermediate_cross_attns = nn.ModuleDict()
            for layer_idx in self.intermediate_cross_attn_layers:
                self.intermediate_cross_attns[f'a2b_{layer_idx}'] = IntermediateCrossAttention(embed_dim)
                self.intermediate_cross_attns[f'b2a_{layer_idx}'] = IntermediateCrossAttention(embed_dim)
            self.stream_merge = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

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

    def prepare_tokens(self, x, modality_masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x, modality_masks=modality_masks)  # [B, N, d]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=False, modality_masks=None):
        """
        Args:
            x: [B, 4, H, W] input tensor (RGB 3ch + IR 1ch)
            return_all_tokens: if True, return all patch tokens
            modality_masks: [B, 2] binary mask for gated cross-attention fusion.

        Returns:
            If return_all_tokens=True: [B, N+1, d] (cls + patch tokens)
            Else: [B, d] (cls token only)
        """
        # Dual-stream path (legacy intermediate cross-attention)
        if (self.intermediate_cross_attn_layers
                and isinstance(self.patch_embed, DualModalPatchEmbed)):
            return self._forward_dual_stream(x, return_all_tokens, modality_masks)

        # Two-stage path: Stage1 (independent) → fusion → Stage2 (fused)
        if self._use_two_stage:
            return self._forward_two_stage(x, return_all_tokens, modality_masks)

        # Original single-stream path (fusion_start_block=0)
        x = self.prepare_tokens(x, modality_masks=modality_masks)
        for blk in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = grad_checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)

        if return_all_tokens:
            return x  # [B, N+1, d] -- cls_token at index 0, patch tokens at 1:
        return x[:, 0]  # [B, d] -- cls token only

    def _add_cls_and_pos(self, tok, w, h, cls_token=None):
        """Add CLS token and positional encoding to patch tokens.

        Args:
            tok: [B, N, d] patch tokens
            w, h: original image width and height
            cls_token: optional CLS token parameter (defaults to self.cls_token)
        Returns:
            [B, N+1, d] with CLS token prepended and positional encoding added
        """
        B = tok.shape[0]
        if cls_token is None:
            cls_token = self.cls_token
        cls = cls_token.expand(B, -1, -1)
        tok = torch.cat((cls, tok), dim=1)  # [B, N+1, d]
        tok = tok + self.interpolate_pos_encoding(tok, w, h)
        return self.pos_drop(tok)

    def _forward_two_stage(self, x, return_all_tokens=False, modality_masks=None):
        """Two-stage forward: Stage1 independent per-modality, Stage2 fused.

        Stage1 (blocks 0..fusion_start_block-1):
            RGB and IR tokens go through the SAME blocks independently.
        Fusion:
            GatedCrossAttention merges rgb and ir streams.
        Stage2 (blocks fusion_start_block..end):
            Fused tokens go through remaining blocks.
        """
        B, nc, w, h = x.shape

        # PatchEmbed: get unfused (rgb_tok, ir_tok)
        rgb_tok, ir_tok = self.patch_embed.forward_dual_stream(x, modality_masks)

        # Add modality-specific CLS tokens + positional encoding
        rgb_tok = self._add_cls_and_pos(rgb_tok, w, h, cls_token=self.cls_token_rgb)
        ir_tok = self._add_cls_and_pos(ir_tok, w, h, cls_token=self.cls_token_ir)

        # Stage1: independent forward through shared blocks
        fsb = self.fusion_start_block
        for blk in self.blocks[:fsb]:
            if self.use_gradient_checkpointing and self.training:
                rgb_tok = grad_checkpoint(blk, rgb_tok, use_reentrant=False)
                ir_tok = grad_checkpoint(blk, ir_tok, use_reentrant=False)
            else:
                rgb_tok = blk(rgb_tok)
                ir_tok = blk(ir_tok)

        # Cache Stage1 outputs for external CLIP projector (only during training)
        # Teacher runs in no_grad/eval mode — skip cache to save memory
        if self.training:
            self._stage1_rgb = rgb_tok  # [B, N+1, d]
            self._stage1_ir = ir_tok

        # Fusion: GatedCrossAttention on patch tokens (exclude CLS for fusion)
        rgb_patches = rgb_tok[:, 1:]  # [B, N, d]
        ir_patches = ir_tok[:, 1:]    # [B, N, d]
        fused_patches = self.mid_fusion(rgb_patches, ir_patches, modality_masks)

        # Re-attach CLS token (average of both streams' CLS)
        fused_cls = 0.5 * (rgb_tok[:, :1] + ir_tok[:, :1])  # [B, 1, d]
        fused = torch.cat([fused_cls, fused_patches], dim=1)  # [B, N+1, d]

        # Stage2: fused forward through remaining blocks
        for blk in self.blocks[fsb:]:
            if self.use_gradient_checkpointing and self.training:
                fused = grad_checkpoint(blk, fused, use_reentrant=False)
            else:
                fused = blk(fused)
        fused = self.norm(fused)

        if return_all_tokens:
            return fused
        return fused[:, 0]

    def _forward_dual_stream(self, x, return_all_tokens=False, modality_masks=None):
        """Dual-stream forward: RGB-dominant and IR-dominant streams with
        intermediate cross-attention at specified layers, merged at the end."""
        B, nc, w, h = x.shape
        rgb_tok, ir_tok = self.patch_embed.forward_dual_stream(x, modality_masks)

        # Each stream gets its own CLS token + positional encoding
        cls_a = self.cls_token.expand(B, -1, -1)
        cls_b = self.cls_token.expand(B, -1, -1)
        stream_a = torch.cat((cls_a, rgb_tok), dim=1)  # RGB-dominant
        stream_b = torch.cat((cls_b, ir_tok), dim=1)    # IR-dominant

        pos = self.interpolate_pos_encoding(stream_a, w, h)
        stream_a = self.pos_drop(stream_a + pos)
        stream_b = self.pos_drop(stream_b + pos)

        for i, blk in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                stream_a = grad_checkpoint(blk, stream_a, use_reentrant=False)
                stream_b = grad_checkpoint(blk, stream_b, use_reentrant=False)
            else:
                stream_a = blk(stream_a)
                stream_b = blk(stream_b)

            # Cross-attention at specified layers
            if i in self.intermediate_cross_attn_layers:
                a2b = self.intermediate_cross_attns[f'a2b_{i}']
                b2a = self.intermediate_cross_attns[f'b2a_{i}']
                stream_a_new = a2b(stream_a, stream_b)
                stream_b_new = b2a(stream_b, stream_a)
                stream_a = stream_a_new
                stream_b = stream_b_new

        stream_a = self.norm(stream_a)
        stream_b = self.norm(stream_b)

        # Merge two streams: concat along feature dim then project
        merged = self.stream_merge(torch.cat([stream_a, stream_b], dim=-1))

        if return_all_tokens:
            return merged
        return merged[:, 0]

    def get_intermediate_layers(self, x, n=1, modality_masks=None):
        """Get outputs from the last n transformer blocks."""
        if self._use_two_stage:
            return self._get_intermediate_layers_two_stage(x, n, modality_masks)
        x = self.prepare_tokens(x, modality_masks=modality_masks)
        output = []
        for i, blk in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                x = grad_checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def _get_intermediate_layers_two_stage(self, x, n=1, modality_masks=None):
        """get_intermediate_layers for two-stage mode."""
        B, nc, w, h = x.shape
        rgb_tok, ir_tok = self.patch_embed.forward_dual_stream(x, modality_masks)
        rgb_tok = self._add_cls_and_pos(rgb_tok, w, h, cls_token=self.cls_token_rgb)
        ir_tok = self._add_cls_and_pos(ir_tok, w, h, cls_token=self.cls_token_ir)

        fsb = self.fusion_start_block
        for blk in self.blocks[:fsb]:
            if self.use_gradient_checkpointing and self.training:
                rgb_tok = grad_checkpoint(blk, rgb_tok, use_reentrant=False)
                ir_tok = grad_checkpoint(blk, ir_tok, use_reentrant=False)
            else:
                rgb_tok = blk(rgb_tok)
                ir_tok = blk(ir_tok)

        rgb_patches = rgb_tok[:, 1:]
        ir_patches = ir_tok[:, 1:]
        fused_patches = self.mid_fusion(rgb_patches, ir_patches, modality_masks)
        fused_cls = 0.5 * (rgb_tok[:, :1] + ir_tok[:, :1])
        fused = torch.cat([fused_cls, fused_patches], dim=1)

        output = []
        stage2_blocks = self.blocks[fsb:]
        for i, blk in enumerate(stage2_blocks):
            if self.use_gradient_checkpointing and self.training:
                fused = grad_checkpoint(blk, fused, use_reentrant=False)
            else:
                fused = blk(fused)
            if len(stage2_blocks) - i <= n:
                output.append(self.norm(fused))
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
