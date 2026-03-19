"""
CLIP-style cross-modal projector for RGB-IR alignment.

Projects Stage1 single-modality features into a shared alignment space
and computes symmetric InfoNCE loss with momentum queues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class CrossModalProjector(nn.Module):
    """CLIP-style dual-tower projector for cross-modal alignment.

    Takes mean-pooled Stage1 patch tokens from RGB-only and IR-only streams,
    projects them into a shared low-dimensional space, and computes symmetric
    InfoNCE contrastive loss with momentum queues for hard negatives.

    Args:
        in_dim: input feature dimension (embed_dim from ViT)
        hidden_dim: hidden layer dimension in MLP projector
        out_dim: output projection dimension
        queue_size: momentum queue size for negative samples
        init_temperature: initial learnable temperature for InfoNCE
    """

    def __init__(self, in_dim=384, hidden_dim=384, out_dim=128,
                 queue_size=16384, init_temperature=0.07):
        super().__init__()
        self.out_dim = out_dim
        self.queue_size = queue_size

        # Separate MLP projectors for each modality (not shared)
        self.rgb_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ir_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Learnable temperature (CLIP-style)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature)))

        # Momentum queues for negative samples
        self.register_buffer(
            'rgb_queue', F.normalize(torch.randn(queue_size, out_dim), dim=-1))
        self.register_buffer(
            'ir_queue', F.normalize(torch.randn(queue_size, out_dim), dim=-1))
        self.register_buffer(
            'queue_ptr', torch.zeros(1, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        for proj in [self.rgb_proj, self.ir_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, rgb_z, ir_z):
        """Update momentum queues with new features (DDP-aware)."""
        # Gather from all GPUs if distributed
        if dist.is_available() and dist.is_initialized():
            rgb_gathered = [torch.zeros_like(rgb_z)
                            for _ in range(dist.get_world_size())]
            ir_gathered = [torch.zeros_like(ir_z)
                           for _ in range(dist.get_world_size())]
            dist.all_gather(rgb_gathered, rgb_z.contiguous())
            dist.all_gather(ir_gathered, ir_z.contiguous())
            rgb_z = torch.cat(rgb_gathered, dim=0)
            ir_z = torch.cat(ir_gathered, dim=0)

        batch_size = rgb_z.shape[0]
        ptr = int(self.queue_ptr)

        # Handle case where batch doesn't fit evenly
        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            self.rgb_queue[ptr:] = rgb_z[:remaining]
            self.ir_queue[ptr:] = ir_z[:remaining]
            overflow = batch_size - remaining
            if overflow > 0:
                self.rgb_queue[:overflow] = rgb_z[remaining:]
                self.ir_queue[:overflow] = ir_z[remaining:]
            self.queue_ptr[0] = overflow
        else:
            self.rgb_queue[ptr:ptr + batch_size] = rgb_z
            self.ir_queue[ptr:ptr + batch_size] = ir_z
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, rgb_feat, ir_feat):
        """Compute symmetric InfoNCE loss with queue negatives.

        Args:
            rgb_feat: [B, in_dim] mean-pooled RGB Stage1 patch tokens
            ir_feat:  [B, in_dim] mean-pooled IR Stage1 patch tokens

        Returns:
            loss: scalar, symmetric InfoNCE loss
        """
        # Project and normalize
        rgb_z = F.normalize(self.rgb_proj(rgb_feat), dim=-1)  # [B, out_dim]
        ir_z = F.normalize(self.ir_proj(ir_feat), dim=-1)     # [B, out_dim]

        # Learnable temperature
        temperature = self.log_temperature.exp().clamp(0.01, 1.0)

        # Positive logits: diagonal (matching pairs)
        # Negative logits: off-diagonal + queue
        ir_all = torch.cat([ir_z, self.ir_queue.clone().detach()], dim=0)
        rgb_all = torch.cat([rgb_z, self.rgb_queue.clone().detach()], dim=0)

        # RGB → IR: each RGB query matches its corresponding IR
        logits_r2i = rgb_z @ ir_all.t() / temperature   # [B, B+Q]
        # IR → RGB: each IR query matches its corresponding RGB
        logits_i2r = ir_z @ rgb_all.t() / temperature   # [B, B+Q]

        labels = torch.arange(rgb_z.shape[0], device=rgb_z.device)

        loss = 0.5 * (F.cross_entropy(logits_r2i, labels) +
                       F.cross_entropy(logits_i2r, labels))

        # Update queues (no grad)
        self._dequeue_and_enqueue(rgb_z.detach(), ir_z.detach())

        return loss
