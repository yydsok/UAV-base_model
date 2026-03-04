"""
Temporal attention module for video frame sequences.
Used in phase 2 when RGB video data is included in pretraining.
When num_frames=1 (single image), this module is automatically bypassed.

References:
- SkySense++ factorized spatio-temporal architecture
- DINO-world temporal frame sampling
- Temporal DINO teacher-student temporal distillation
"""

import math
import torch
import torch.nn as nn


class TemporalPositionEncoding(nn.Module):
    """
    Encode absolute timestamps (in seconds) for each video frame.
    Supports both learnable and sinusoidal encoding.
    """
    def __init__(self, embed_dim, max_frames=64, encoding_type='learnable'):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type

        if encoding_type == 'learnable':
            self.temporal_embed = nn.Parameter(
                torch.zeros(1, max_frames, embed_dim))
            nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        elif encoding_type == 'sinusoidal':
            # Sinusoidal encoding based on absolute timestamps
            self.register_buffer('_dummy', torch.empty(0))
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _sinusoidal_encode(self, timestamps, dim):
        """
        Args:
            timestamps: [B, T] absolute timestamps in seconds
            dim: embedding dimension
        Returns:
            [B, T, dim] temporal embeddings
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestamps.device) * -emb)
        emb = timestamps.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :, :1])], dim=-1)
        return emb

    def forward(self, num_frames, timestamps=None):
        """
        Args:
            num_frames: int, number of frames
            timestamps: optional [B, T] absolute timestamps in seconds
        Returns:
            [1, T, d] or [B, T, d] temporal position encoding
        """
        if self.encoding_type == 'learnable':
            return self.temporal_embed[:, :num_frames, :]
        else:
            if timestamps is None:
                # Default: equally spaced frame indices
                timestamps = torch.arange(
                    num_frames, dtype=torch.float32,
                    device=self._dummy.device).unsqueeze(0)
            return self._sinusoidal_encode(timestamps, self.embed_dim)


class TemporalAttention(nn.Module):
    """
    Cross-frame temporal attention layer.
    Operates on spatial features from multiple frames.

    Architecture: After spatial ViT encoding, apply temporal attention
    across frames at each spatial position.
    """
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0,
                 drop=0., attn_drop=0., num_layers=2, max_frames=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_pos_enc = TemporalPositionEncoding(
            embed_dim, max_frames=max_frames, encoding_type='learnable')

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TemporalAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, num_frames, timestamps=None):
        """
        Args:
            x: [B*T, N, d] spatial features from ViT backbone
               B=batch, T=num_frames, N=num_patches
            num_frames: int, number of frames per sample
            timestamps: optional [B, T] frame timestamps

        Returns:
            [B*T, N, d] temporally-enhanced features

        When num_frames=1 (single image), returns input unchanged.
        """
        if num_frames <= 1:
            return x  # Bypass for single images

        BT, N, d = x.shape
        B = BT // num_frames
        T = num_frames

        # Reshape: [B*T, N, d] -> [B*N, T, d] for temporal attention
        x = x.reshape(B, T, N, d)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, d)

        # Add temporal position encoding
        temp_pos = self.temporal_pos_enc(T, timestamps)  # [1, T, d]
        x = x + temp_pos

        # Apply temporal attention blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Reshape back: [B*N, T, d] -> [B*T, N, d]
        x = x.reshape(B, N, T, d)
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, d)

        return x


class TemporalAttentionBlock(nn.Module):
    """Single temporal attention block with self-attention + FFN."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # Self-attention across temporal dimension
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class VideoFrameSampler:
    """
    Sample video frames with randomized temporal intervals.
    Reference: DINO-world uniform temporal increment sampling.

    Strategy: randomly sample delta_t from a predefined range, accumulate
    timestamps to get frames at varying temporal distances.
    """
    def __init__(self, num_frames=4, min_delta=1, max_delta=30,
                 fps=30.0):
        """
        Args:
            num_frames: number of frames to sample per clip
            min_delta: minimum frame interval
            max_delta: maximum frame interval
            fps: video frame rate (for timestamp computation)
        """
        self.num_frames = num_frames
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.fps = fps

    def sample(self, total_frames):
        """
        Sample frame indices and compute timestamps.

        Args:
            total_frames: total number of frames in the video

        Returns:
            indices: list of int, sampled frame indices
            timestamps: list of float, absolute timestamps in seconds
        """
        import random
        indices = [random.randint(0, total_frames - 1)]
        for _ in range(self.num_frames - 1):
            delta = random.randint(self.min_delta, self.max_delta)
            next_idx = indices[-1] + delta
            next_idx = min(next_idx, total_frames - 1)
            indices.append(next_idx)

        timestamps = [idx / self.fps for idx in indices]
        return indices, timestamps
