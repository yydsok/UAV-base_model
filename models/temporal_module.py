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
        self.max_frames = max_frames
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
        if half_dim == 0:
            return torch.zeros(*timestamps.shape, dim, device=timestamps.device,
                               dtype=timestamps.dtype)
        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestamps.device) * -emb)
        emb = timestamps.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :, :1])], dim=-1)
        return emb

    def _interpolate_learnable(self, positions):
        """Interpolate learnable temporal embeddings by fractional positions.

        Args:
            positions: [B, T] float indices in [0, max_frames-1]
        Returns:
            [B, T, d]
        """
        table = self.temporal_embed[0]  # [M, d]
        max_idx = table.shape[0] - 1
        pos = positions.clamp(min=0.0, max=float(max_idx))
        idx0 = pos.floor().long()
        idx1 = (idx0 + 1).clamp(max=max_idx)
        w = (pos - idx0.float()).unsqueeze(-1)
        emb0 = table[idx0]  # [B, T, d]
        emb1 = table[idx1]  # [B, T, d]
        return emb0 * (1.0 - w) + emb1 * w

    def forward(self, num_frames, timestamps=None):
        """
        Args:
            num_frames: int, number of frames
            timestamps: optional [B, T] absolute timestamps in seconds
        Returns:
            [1, T, d] or [B, T, d] temporal position encoding
        """
        if self.encoding_type == 'learnable':
            if timestamps is not None:
                # Use relative timestamps and map them to learnable table indices.
                ts = timestamps.to(self.temporal_embed.device)
                if ts.shape[-1] != num_frames:
                    raise ValueError(
                        f"timestamps.shape[-1]={ts.shape[-1]} != num_frames={num_frames}"
                    )
                rel = ts - ts[:, :1]
                span = rel.max(dim=1, keepdim=True).values.clamp(min=1e-6)
                scaled = rel / span * max(1, self.max_frames - 1)
                return self._interpolate_learnable(scaled)

            if num_frames <= self.max_frames:
                return self.temporal_embed[:, :num_frames, :]

            # For long clips, interpolate to avoid length mismatch.
            pos = torch.linspace(
                0.0, float(self.max_frames - 1), steps=num_frames,
                device=self.temporal_embed.device
            ).unsqueeze(0)
            return self._interpolate_learnable(pos)
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
        temp_pos = self.temporal_pos_enc(T, timestamps)  # [1, T, d] or [B, T, d]
        # Fix sinusoidal shape: if [B, T, d] but x is [B*N, T, d], expand
        if temp_pos.shape[0] == B and N > 1:
            temp_pos = temp_pos.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, T, d)
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

        if self.num_frames <= 0:
            return [], []
        if total_frames <= 0:
            return [0] * self.num_frames, [0.0] * self.num_frames

        # Short video protection: if fewer frames than requested,
        # sample what we can and pad with repeated last frame
        if total_frames < self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            timestamps = [idx / self.fps for idx in indices]
            return indices, timestamps

        # Ensure minimum separation so frames don't saturate at end
        min_delta_eff = max(self.min_delta, 3)  # at least 3 frames apart
        min_span = (self.num_frames - 1) * min_delta_eff
        max_start = max(0, total_frames - 1 - min_span)

        indices = [random.randint(0, max_start)]
        for k in range(1, self.num_frames):
            remaining_after = self.num_frames - 1 - k
            # Max value this frame can take, leaving room for remaining frames
            max_val = total_frames - 1 - remaining_after * min_delta_eff
            min_val = indices[-1] + min_delta_eff
            if min_val > max_val:
                min_val = max_val
            upper = min(indices[-1] + self.max_delta, max_val)
            upper = max(upper, min_val)
            indices.append(random.randint(min_val, upper))

        timestamps = [idx / self.fps for idx in indices]
        return indices, timestamps

    def sample_from_available(self, available_indices):
        """
        Sample from a sparse set of available frame indices.
        Used for aerial-rgbt sequences where frames are not contiguous.

        Args:
            available_indices: sorted list of available frame indices

        Returns:
            indices: list of int, sampled frame indices
            timestamps: list of float, absolute timestamps in seconds
        """
        import random
        if self.num_frames <= 0:
            return [], []
        n = len(available_indices)
        if n == 0:
            return [0] * self.num_frames, [0.0] * self.num_frames

        if n >= self.num_frames:
            selected_positions = sorted(random.sample(range(n), self.num_frames))
        else:
            # Pad with repeated last frame
            selected_positions = list(range(n))
            while len(selected_positions) < self.num_frames:
                selected_positions.append(selected_positions[-1])

        indices = [available_indices[p] for p in selected_positions]
        timestamps = [idx / self.fps for idx in indices]
        return indices, timestamps
