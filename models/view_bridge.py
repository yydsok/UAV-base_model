"""
View-Domain Bridge module with Sinkhorn-Knopp Optimal Transport.

Projects image-level features to a shared embedding space with continuous
prototypes updated via EMA. Student predicts soft equipartitioned
assignments computed by Sinkhorn-Knopp from the teacher's features.

Replaces the previous Spherical K-Means hard-clustering approach which
caused loss oscillation every update_interval steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ViewDomainBridge(nn.Module):
    """Project features and produce prototype logits for bridge loss.

    V4 changes (Sinkhorn-Knopp):
    - prototypes: persistent nn.Parameter updated via EMA (no random re-init)
    - Sinkhorn-Knopp computes soft, equipartitioned assignments from teacher
    - No feature queue, no hard clustering, no periodic re-clustering jumps
    """

    def __init__(self, in_dim, proj_dim=256, num_prototypes=64,
                 sinkhorn_iters=3, sinkhorn_temp=0.1, prototype_ema=0.999,
                 **kwargs):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_prototypes = num_prototypes
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temp = sinkhorn_temp
        self.prototype_ema = prototype_ema

        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )
        self.teacher_projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )
        self.teacher_projector.load_state_dict(self.projector.state_dict())
        for p in self.teacher_projector.parameters():
            p.requires_grad = False

        # Learnable prototypes (updated via EMA from teacher features)
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(num_prototypes, proj_dim), dim=-1))
        # Don't include in optimizer — updated via EMA only
        self.prototypes.requires_grad = False

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
        proto = F.normalize(self.prototypes.data, dim=-1)
        logits = z @ proto.t()
        return z, logits

    @torch.no_grad()
    def teacher_forward(self, x):
        """Project teacher features using the EMA bridge projector."""
        z = self.teacher_projector(x)
        z = F.normalize(z, dim=-1)
        proto = F.normalize(self.prototypes.data, dim=-1)
        logits = z @ proto.t()
        return z, logits

    @torch.no_grad()
    def update_teacher_projector(self, momentum):
        """EMA update for the teacher-side bridge projector."""
        for ps, pt in zip(self.projector.parameters(),
                          self.teacher_projector.parameters()):
            pt.data.mul_(momentum).add_((1 - momentum) * ps.detach().data)

    @torch.no_grad()
    def sinkhorn_assignments(self, teacher_z):
        """Compute soft equipartitioned assignments using Sinkhorn-Knopp OT.

        Args:
            teacher_z: [B, proj_dim] L2-normalized teacher projected features

        Returns:
            Q: [B, K] soft assignment matrix (rows sum to 1, columns ~uniform)
        """
        proto = F.normalize(self.prototypes.data, dim=-1)
        logits = teacher_z @ proto.t()  # [B_local, K]
        scaled_logits = (logits / self.sinkhorn_temp).float()

        B_local, K = scaled_logits.shape
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        B_total = B_local * world_size

        # Balanced Sinkhorn becomes degenerate when the batch has fewer samples
        # than prototypes: each sample is forced to spread mass across many
        # prototypes, producing overly high-entropy targets. Fall back to a
        # plain teacher softmax in that regime.
        if B_total < K:
            return F.softmax(scaled_logits, dim=-1).to(teacher_z.dtype)

        scaled_logits = scaled_logits - scaled_logits.max(dim=1, keepdim=True).values
        Q = torch.exp(scaled_logits).t()  # [K, B_local]

        sum_Q = Q.sum()
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q = Q / sum_Q.clamp(min=1e-12)

        for _ in range(self.sinkhorn_iters):
            row_sums = Q.sum(dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(row_sums)
            Q = Q / row_sums.clamp(min=1e-12)
            Q = Q / float(K)

            col_sums = Q.sum(dim=0, keepdim=True)
            Q = Q / col_sums.clamp(min=1e-12)
            Q = Q / float(B_total)

        Q = Q * float(B_total)
        return Q.t().to(teacher_z.dtype)

    @torch.no_grad()
    def update_prototypes_ema(self, teacher_z, assignments=None):
        """Update prototypes via EMA from teacher projected features.

        Args:
            teacher_z: [B, proj_dim] L2-normalized teacher features
            assignments: optional [B, K] Sinkhorn assignments computed with the
                         same target rule used by the bridge loss.
        """
        # Gather from all GPUs if distributed
        if dist.is_initialized():
            gathered = [torch.zeros_like(teacher_z)
                        for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, teacher_z)
            all_z = torch.cat(gathered, dim=0)  # [B*world, proj_dim]
            if assignments is not None:
                gathered_assign = [torch.zeros_like(assignments)
                                   for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_assign, assignments)
                all_assign = torch.cat(gathered_assign, dim=0)
            else:
                all_assign = None
        else:
            all_z = teacher_z
            all_assign = assignments

        # Use the same assignment rule for supervision and prototype updates.
        if all_assign is None:
            all_assign = self.sinkhorn_assignments(all_z)

        # Weighted average of features per prototype
        new_proto = all_assign.t() @ all_z  # [K, proj_dim]
        counts = all_assign.sum(dim=0, keepdim=True).t().clamp(min=1e-6)  # [K, 1]
        new_proto = new_proto / counts
        new_proto = F.normalize(new_proto, dim=-1)

        # EMA update
        self.prototypes.data = (
            self.prototype_ema * F.normalize(self.prototypes.data, dim=-1)
            + (1 - self.prototype_ema) * new_proto
        )
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
