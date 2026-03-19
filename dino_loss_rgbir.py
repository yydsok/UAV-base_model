"""
Multi-modal DINO pretraining loss functions.

Contains:
- DINOLoss: Standard DINO self-distillation loss
- MGCLLoss: Multi-granularity contrastive loss (token/object/image)
- TCLLoss: Legacy time-contrastive learning objective
- PATCLoss: Perspective-adaptive temporal correspondence objective
- ViewInvarianceLoss: Affine viewpoint invariance loss (ATCL, small-angle)
- ViewBridgeLoss: View-domain bridge loss (unsupervised prototype consistency)
- PretrainingLoss: Combined weighted loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINOLoss(nn.Module):
    """Standard DINO self-distillation loss."""

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp,
                           warmup_teacher_temp_epochs),
            torch.ones(max(0, nepochs - warmup_teacher_temp_epochs)) * teacher_temp
        ])

    def forward(self, student_output, teacher_output, epoch, ncrops=None,
                teacher_indices=None, update_center=True):
        """
        Args:
            student_output: [B*ncrops, out_dim]
            teacher_output: [B*2, out_dim] (always 2 anchor views)
            epoch: current epoch
            ncrops: override for self.ncrops (e.g. T for video frames)
        """
        ncrops_eff = ncrops if ncrops is not None else self.ncrops
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(ncrops_eff)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        if len(student_out) == 0:
            raise ValueError("student_output is empty after chunking.")
        batch_size = student_out[0].shape[0]
        if batch_size <= 0:
            raise ValueError("Invalid student chunk batch size.")
        if teacher_output.shape[0] % batch_size != 0:
            raise ValueError(
                f"teacher_output.shape[0]={teacher_output.shape[0]} "
                f"is not divisible by batch_size={batch_size}."
            )
        n_teacher_views = teacher_output.shape[0] // batch_size
        teacher_out = teacher_out.detach().chunk(n_teacher_views)

        if teacher_indices is None:
            teacher_indices = list(range(n_teacher_views))
        if len(teacher_indices) != n_teacher_views:
            raise ValueError(
                f"len(teacher_indices)={len(teacher_indices)} "
                f"!= n_teacher_views={n_teacher_views}"
            )

        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            skip_student_idx = teacher_indices[iq]
            for v in range(len(student_out)):
                if v == skip_student_idx:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        if update_center:
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)

        self.center = (self.center * self.center_momentum
                       + batch_center * (1 - self.center_momentum))


class MGCLLoss(nn.Module):
    """Multi-Granularity Contrastive Learning loss."""

    def __init__(self, proj_dim=256, temperature=0.1, center_momentum=0.9):
        super().__init__()
        self.temperature = temperature
        self.center_momentum = center_momentum
        self.register_buffer("center_token", torch.zeros(1, proj_dim))
        self.register_buffer("center_object", torch.zeros(1, proj_dim))
        self.register_buffer("center_image", torch.zeros(1, proj_dim))

    def _contrastive_loss(self, student_feat, teacher_feat, center):
        if student_feat.dim() == 3:
            bsz, n_tokens, dim = student_feat.shape
            student_feat = student_feat.reshape(bsz * n_tokens, dim)
            teacher_feat = teacher_feat.reshape(bsz * n_tokens, dim)

        # Keep MGCL logits on a bounded angular scale. Unlike DINOHead, these
        # projection MLPs do not have a normalized classifier layer, so raw
        # feature magnitudes can otherwise grow and make the distillation loss
        # explode even when directions are still reasonable.
        student_feat = F.normalize(student_feat, dim=-1, eps=1e-6)
        teacher_feat = F.normalize(teacher_feat, dim=-1, eps=1e-6)

        s = F.log_softmax(student_feat / self.temperature, dim=-1)
        with torch.no_grad():
            t = F.softmax((teacher_feat - center) / (self.temperature * 0.4), dim=-1)
        return -torch.sum(t * s, dim=-1).mean()

    def forward(self, student_mg, teacher_mg, update_center=True):
        token_loss = self._contrastive_loss(student_mg['token'], teacher_mg['token'], self.center_token)
        object_loss = self._contrastive_loss(student_mg['object'], teacher_mg['object'], self.center_object)
        image_loss = self._contrastive_loss(student_mg['image'], teacher_mg['image'], self.center_image)
        total = (token_loss + object_loss + image_loss) / 3.0

        if update_center:
            self.update_centers(teacher_mg)

        return {
            'token_loss': token_loss,
            'object_loss': object_loss,
            'image_loss': image_loss,
            'total': total,
        }

    @torch.no_grad()
    def update_centers(self, teacher_mg):
        self.update_centers_multi([teacher_mg])

    @torch.no_grad()
    def update_centers_multi(self, teacher_mg_list):
        self._update_center_multi(
            self.center_token, [teacher_mg['token'] for teacher_mg in teacher_mg_list]
        )
        self._update_center_multi(
            self.center_object, [teacher_mg['object'] for teacher_mg in teacher_mg_list]
        )
        self._update_center_multi(
            self.center_image, [teacher_mg['image'] for teacher_mg in teacher_mg_list]
        )

    @torch.no_grad()
    def _update_center(self, center, teacher_feat):
        if teacher_feat.dim() == 3:
            teacher_feat = F.normalize(teacher_feat, dim=-1, eps=1e-6)
            batch_center = teacher_feat.mean(dim=[0, 1], keepdim=False).unsqueeze(0)
        elif teacher_feat.dim() == 2:
            teacher_feat = F.normalize(teacher_feat, dim=-1, eps=1e-6)
            batch_center = teacher_feat.mean(dim=0, keepdim=True)
        else:
            return

        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        center.data = (center.data * self.center_momentum
                       + batch_center * (1 - self.center_momentum))

    @torch.no_grad()
    def _update_center_multi(self, center, teacher_feats):
        total_sum = None
        total_count = None

        for teacher_feat in teacher_feats:
            if teacher_feat.dim() == 3:
                teacher_feat = F.normalize(teacher_feat, dim=-1, eps=1e-6)
                feat_sum = teacher_feat.sum(dim=[0, 1], keepdim=False).unsqueeze(0)
                feat_count = torch.tensor(
                    float(teacher_feat.shape[0] * teacher_feat.shape[1]),
                    device=teacher_feat.device,
                    dtype=teacher_feat.dtype,
                )
            elif teacher_feat.dim() == 2:
                teacher_feat = F.normalize(teacher_feat, dim=-1, eps=1e-6)
                feat_sum = teacher_feat.sum(dim=0, keepdim=True)
                feat_count = torch.tensor(
                    float(teacher_feat.shape[0]),
                    device=teacher_feat.device,
                    dtype=teacher_feat.dtype,
                )
            else:
                continue

            total_sum = feat_sum if total_sum is None else total_sum + feat_sum
            total_count = feat_count if total_count is None else total_count + feat_count

        if total_sum is None or total_count is None:
            return

        if dist.is_initialized():
            dist.all_reduce(total_sum)
            dist.all_reduce(total_count)

        batch_center = total_sum / total_count.clamp(min=1e-6)
        center.data = (center.data * self.center_momentum
                       + batch_center * (1 - self.center_momentum))



class TCLLoss(nn.Module):
    """Time-Contrastive Learning loss.

    Global: InfoNCE over pooled frame features.
    Patch: Cross-frame spatial correspondence with teacher soft targets
           and Gaussian spatial prior.
    """

    loss_name = "tcl"
    patch_loss_name = "tcl_patch"

    def __init__(self, temperature=0.07, patch_temperature=0.1,
                 spatial_sigma_base=2.0, spatial_sigma_scale=0.5):
        super().__init__()
        self.temperature = temperature
        self.patch_temperature = patch_temperature
        self.spatial_sigma_base = spatial_sigma_base
        self.spatial_sigma_scale = spatial_sigma_scale

    @staticmethod
    def _zero_like(reference):
        return torch.tensor(0.0, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _validate_patch_shapes(student_patches, teacher_patches, num_frames, grid_size):
        BT, N, d = student_patches.shape
        if teacher_patches.shape != student_patches.shape:
            raise ValueError(
                f"teacher_patches.shape={teacher_patches.shape} "
                f"!= student_patches.shape={student_patches.shape}"
            )
        if BT % num_frames != 0:
            raise ValueError(f"BT={BT} is not divisible by num_frames={num_frames}")
        if grid_size * grid_size != N:
            raise ValueError(
                f"grid_size²={grid_size**2} != N={N}. "
                f"Ensure grid_size matches patch grid (e.g. 16 for 224/14)."
            )
        return BT, N, d

    @staticmethod
    def _build_patch_positions(grid_size, device, dtype):
        return torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=dtype),
            torch.arange(grid_size, device=device, dtype=dtype),
            indexing='ij'
        ), dim=-1).reshape(grid_size * grid_size, 2)

    @staticmethod
    def _normalize_rows(prior):
        return prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _pair_gap(batch_size, ti, tj, device, dtype, timestamps=None, expected_shape=None):
        if timestamps is None:
            return torch.full((batch_size,), float(abs(tj - ti)), device=device, dtype=dtype)
        if expected_shape is not None and timestamps.shape != expected_shape:
            raise ValueError(f"timestamps.shape={timestamps.shape}, expected {expected_shape}")
        return (timestamps[:, tj] - timestamps[:, ti]).abs().to(device=device, dtype=dtype)

    def _identity_prior(self, spatial_dist_sq, sigma):
        sigma = sigma.clamp(min=1e-6)
        prior = torch.exp(
            -spatial_dist_sq.unsqueeze(0) /
            (2.0 * sigma[:, None, None] * sigma[:, None, None])
        )
        return self._normalize_rows(prior)

    def forward(self, frame_features, num_frames, timestamps=None):
        """
        Args:
            frame_features: [B*T, d] pooled features per frame
            num_frames: int T (>=2)
            timestamps: unused in vanilla TCL, kept for API compatibility
        Returns:
            scalar loss
        """
        del timestamps
        if num_frames <= 1:
            return self._zero_like(frame_features)

        BT, d = frame_features.shape
        B = BT // num_frames
        T = num_frames

        # L2 normalize
        feats = F.normalize(frame_features, dim=-1, eps=1e-8)

        # No cross-sample negatives when B=1; fallback to adjacent-frame
        # consistency to keep gradients non-zero.
        if B <= 1:
            feats_bt = feats.reshape(B, T, d)
            if T <= 1:
                return self._zero_like(frame_features)
            pos_sim = (feats_bt[:, 1:] * feats_bt[:, :-1]).sum(dim=-1)
            return (1.0 - pos_sim).mean()

        # Similarity matrix [BT, BT]
        sim = feats @ feats.t() / self.temperature
        sim = sim.clamp(min=-100, max=100)

        # Positive mask: same-sequence frame pairs (block-diagonal, exclude self)
        seq_ids = torch.arange(B, device=feats.device).unsqueeze(1).expand(B, T).reshape(BT)
        pos_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0))  # [BT, BT]
        self_mask = torch.eye(BT, device=feats.device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask  # exclude self

        # InfoNCE: for each anchor, log(sum_pos_exp / sum_all_exp)
        neg_mask = ~self_mask
        mask_fill = torch.finfo(sim.dtype).min
        log_denom = torch.logsumexp(sim.masked_fill(~neg_mask, mask_fill), dim=1)  # [BT]
        log_numer = torch.logsumexp(sim.masked_fill(~pos_mask, mask_fill), dim=1)  # [BT]
        return -(log_numer - log_denom).mean()

    def forward_patch(self, student_patches, teacher_patches, num_frames, grid_size,
                      timestamps=None):
        """Patch-level temporal correspondence loss."""
        if num_frames <= 1:
            return self._zero_like(student_patches)

        BT, N, d = self._validate_patch_shapes(
            student_patches, teacher_patches, num_frames, grid_size)
        B = BT // num_frames
        T = num_frames

        s_patches = student_patches.reshape(B, T, N, d)
        t_patches = teacher_patches.reshape(B, T, N, d)
        positions = self._build_patch_positions(
            grid_size, student_patches.device, student_patches.dtype)
        spatial_dist_sq = ((positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2).sum(-1)

        loss = 0.0
        n_pairs = 0
        for ti in range(T):
            for tj in range(ti + 1, T):
                dt = self._pair_gap(
                    B, ti, tj, student_patches.device, student_patches.dtype,
                    timestamps=timestamps, expected_shape=(B, T) if timestamps is not None else None)
                sigma = self.spatial_sigma_base + self.spatial_sigma_scale * dt
                spatial_prior = self._identity_prior(spatial_dist_sq, sigma)

                with torch.no_grad():
                    t_i = F.normalize(t_patches[:, ti], dim=-1)
                    t_j = F.normalize(t_patches[:, tj], dim=-1)
                    teacher_sim = torch.bmm(t_i, t_j.transpose(1, 2))
                    teacher_sim = (teacher_sim / self.patch_temperature).clamp(-100, 100)
                    log_prior = torch.log(spatial_prior.clamp(min=1e-8))
                    teacher_target = F.softmax(teacher_sim + log_prior, dim=-1)

                s_i = F.normalize(s_patches[:, ti], dim=-1)
                s_j = F.normalize(s_patches[:, tj], dim=-1)
                student_sim = torch.bmm(s_i, s_j.transpose(1, 2))
                student_log_prob = F.log_softmax(
                    (student_sim / self.patch_temperature).clamp(-100, 100), dim=-1)

                pair_loss = -(teacher_target * student_log_prob).sum(dim=-1).mean()
                loss = loss + pair_loss
                n_pairs += 1

        return loss / max(1, n_pairs)


class PATCLoss(TCLLoss):
    """Perspective-Adaptive Temporal Correspondence.

    Extends vanilla TCL with:
      1. Geometry-guided correspondence prior from teacher-induced patch anchors
      2. Confidence-aware filtering via teacher entropy
      3. View-gap adaptive weighting that emphasizes harder wide-baseline pairs
    """

    loss_name = "patc"
    patch_loss_name = "patc_patch"

    def __init__(self, temperature=0.07, patch_temperature=0.1,
                 spatial_sigma_base=2.0, spatial_sigma_scale=0.5,
                 geometry_blend=0.7, confidence_power=1.5,
                 min_confidence=0.05, gap_weight_scale=0.5):
        super().__init__(
            temperature=temperature,
            patch_temperature=patch_temperature,
            spatial_sigma_base=spatial_sigma_base,
            spatial_sigma_scale=spatial_sigma_scale,
        )
        self.geometry_blend = geometry_blend
        self.confidence_power = confidence_power
        self.min_confidence = min_confidence
        self.gap_weight_scale = gap_weight_scale

    def _gap_weights(self, gaps):
        if gaps.numel() == 0:
            return gaps
        denom = gaps.max().clamp(min=1.0)
        return 1.0 + self.gap_weight_scale * (gaps / denom)

    def _geometry_prior(self, positions, expected_positions, sigma):
        sigma = sigma.clamp(min=1e-6)
        target_positions = positions.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        dist_sq = ((target_positions - expected_positions.unsqueeze(2)) ** 2).sum(dim=-1)
        prior = torch.exp(
            -dist_sq / (2.0 * sigma[:, None, None] * sigma[:, None, None])
        )
        return self._normalize_rows(prior)

    def forward(self, frame_features, num_frames, timestamps=None):
        if num_frames <= 1:
            return self._zero_like(frame_features)

        BT, d = frame_features.shape
        B = BT // num_frames
        T = num_frames
        feats = F.normalize(frame_features, dim=-1, eps=1e-8)

        if B <= 1:
            feats_bt = feats.reshape(B, T, d)
            if T <= 1:
                return self._zero_like(frame_features)
            pos_sim = (feats_bt[:, 1:] * feats_bt[:, :-1]).sum(dim=-1)
            gaps = self._pair_gap(
                B, 0, 1, feats.device, feats.dtype,
                timestamps=timestamps[:, :2] if timestamps is not None and T > 1 else None,
                expected_shape=(B, 2) if timestamps is not None and T > 1 else None)
            weights = self._gap_weights(gaps).unsqueeze(1).expand_as(pos_sim)
            return ((1.0 - pos_sim) * weights).sum() / weights.sum().clamp(min=1e-6)

        sim = (feats @ feats.t() / self.temperature).clamp(min=-100, max=100)

        seq_ids = torch.arange(B, device=feats.device).unsqueeze(1).expand(B, T).reshape(BT)
        frame_ids = torch.arange(T, device=feats.device).unsqueeze(0).expand(B, T).reshape(BT)

        pos_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0))
        self_mask = torch.eye(BT, device=feats.device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask
        neg_mask = ~self_mask

        if timestamps is not None:
            if timestamps.shape != (B, T):
                raise ValueError(f"timestamps.shape={timestamps.shape}, expected {(B, T)}")
            flat_time = timestamps.to(device=feats.device, dtype=feats.dtype).reshape(BT)
            gaps = (flat_time.unsqueeze(1) - flat_time.unsqueeze(0)).abs()
        else:
            gaps = (frame_ids.unsqueeze(1) - frame_ids.unsqueeze(0)).abs().to(sim.dtype)

        pos_weights = self._gap_weights(gaps).clamp(min=1e-6)

        mask_fill = torch.finfo(sim.dtype).min
        log_denom = torch.logsumexp(sim.masked_fill(~neg_mask, mask_fill), dim=1)
        # Apply gap weights symmetrically: weight both numer and denom
        # to avoid systematic negative bias from log(pos_weights)
        log_numer = torch.logsumexp(sim.masked_fill(~pos_mask, mask_fill), dim=1)
        # Gap-weighted average: weight the per-sample loss, not the logits
        per_sample_loss = -(log_numer - log_denom)  # standard InfoNCE per sample
        # Weight by mean gap weight per sample (larger gaps = more important)
        sample_gap_weight = (pos_weights * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1)
        return (per_sample_loss * sample_gap_weight).mean() / sample_gap_weight.mean().clamp(min=1e-6)

    def forward_patch(self, student_patches, teacher_patches, num_frames, grid_size,
                      timestamps=None):
        if num_frames <= 1:
            return self._zero_like(student_patches)

        BT, N, d = self._validate_patch_shapes(
            student_patches, teacher_patches, num_frames, grid_size)
        B = BT // num_frames
        T = num_frames

        s_patches = student_patches.reshape(B, T, N, d)
        t_patches = teacher_patches.reshape(B, T, N, d)
        positions = self._build_patch_positions(
            grid_size, student_patches.device, student_patches.dtype)
        spatial_dist_sq = ((positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2).sum(-1)

        loss = student_patches.new_tensor(0.0)
        total_weight = student_patches.new_tensor(0.0)
        max_entropy = math.log(max(N, 2))

        for ti in range(T):
            for tj in range(ti + 1, T):
                dt = self._pair_gap(
                    B, ti, tj, student_patches.device, student_patches.dtype,
                    timestamps=timestamps, expected_shape=(B, T) if timestamps is not None else None)
                sigma = self.spatial_sigma_base + self.spatial_sigma_scale * dt
                identity_prior = self._identity_prior(spatial_dist_sq, sigma)
                pair_weights = self._gap_weights(dt)  # [B]

                with torch.no_grad():
                    t_i = F.normalize(t_patches[:, ti], dim=-1)
                    t_j = F.normalize(t_patches[:, tj], dim=-1)
                    teacher_sim = torch.bmm(t_i, t_j.transpose(1, 2))
                    teacher_logits = (teacher_sim / self.patch_temperature).clamp(-100, 100)

                    coarse_teacher = F.softmax(teacher_logits, dim=-1)
                    expected_positions = coarse_teacher @ positions  # [B, N, 2]
                    geometry_prior = self._geometry_prior(positions, expected_positions, sigma)

                    guided_prior = (
                        (1.0 - self.geometry_blend) * identity_prior
                        + self.geometry_blend * geometry_prior
                    )
                    guided_prior = self._normalize_rows(guided_prior)
                    teacher_target = F.softmax(
                        teacher_logits + torch.log(guided_prior.clamp(min=1e-8)), dim=-1)

                    entropy = -(teacher_target * torch.log(teacher_target.clamp(min=1e-8))).sum(dim=-1)
                    confidence = 1.0 - entropy / max_entropy
                    confidence = confidence.clamp(min=self.min_confidence).pow(self.confidence_power)

                s_i = F.normalize(s_patches[:, ti], dim=-1)
                s_j = F.normalize(s_patches[:, tj], dim=-1)
                student_sim = torch.bmm(s_i, s_j.transpose(1, 2))
                student_log_prob = F.log_softmax(
                    (student_sim / self.patch_temperature).clamp(-100, 100), dim=-1)

                per_patch_loss = -(teacher_target * student_log_prob).sum(dim=-1)  # [B, N]
                patch_weights = confidence * pair_weights[:, None]
                loss = loss + (per_patch_loss * patch_weights).sum()
                total_weight = total_weight + patch_weights.sum()

        return loss / total_weight.clamp(min=1e-6)


class CrossModalPatchLoss(nn.Module):
    """Teacher-guided RGB patch <-> IR patch soft correspondence loss."""

    def __init__(self, patch_temperature=0.1, spatial_sigma=1.5,
                 geometry_blend=0.7, confidence_power=1.5,
                 min_confidence=0.05):
        super().__init__()
        self.patch_temperature = patch_temperature
        self.spatial_sigma = spatial_sigma
        self.geometry_blend = geometry_blend
        self.confidence_power = confidence_power
        self.min_confidence = min_confidence

    @staticmethod
    def _zero_like(reference):
        return torch.tensor(0.0, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _build_patch_positions(grid_size, device, dtype):
        return torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=dtype),
            torch.arange(grid_size, device=device, dtype=dtype),
            indexing='ij'
        ), dim=-1).reshape(grid_size * grid_size, 2)

    @staticmethod
    def _normalize_rows(prior):
        return prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def _identity_prior(self, spatial_dist_sq, sigma):
        sigma = sigma.clamp(min=1e-6)
        prior = torch.exp(
            -spatial_dist_sq.unsqueeze(0) /
            (2.0 * sigma[:, None, None] * sigma[:, None, None])
        )
        return self._normalize_rows(prior)

    def _geometry_prior(self, positions, expected_positions, sigma):
        sigma = sigma.clamp(min=1e-6)
        target_positions = positions.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        dist_sq = ((target_positions - expected_positions.unsqueeze(2)) ** 2).sum(dim=-1)
        prior = torch.exp(
            -dist_sq / (2.0 * sigma[:, None, None] * sigma[:, None, None])
        )
        return self._normalize_rows(prior)

    def _teacher_targets(self, anchor_patches, target_patches, positions, identity_prior, sigma):
        max_entropy = math.log(max(anchor_patches.shape[1], 2))
        teacher_sim = torch.bmm(anchor_patches, target_patches.transpose(1, 2))
        teacher_logits = (teacher_sim / self.patch_temperature).clamp(-100, 100)
        coarse_teacher = F.softmax(teacher_logits, dim=-1)
        expected_positions = coarse_teacher @ positions
        geometry_prior = self._geometry_prior(positions, expected_positions, sigma)
        guided_prior = (
            (1.0 - self.geometry_blend) * identity_prior
            + self.geometry_blend * geometry_prior
        )
        guided_prior = self._normalize_rows(guided_prior)
        teacher_target = F.softmax(
            teacher_logits + torch.log(guided_prior.clamp(min=1e-8)), dim=-1)
        entropy = -(teacher_target * torch.log(teacher_target.clamp(min=1e-8))).sum(dim=-1)
        confidence = 1.0 - entropy / max_entropy
        confidence = confidence.clamp(min=self.min_confidence).pow(self.confidence_power)
        return teacher_target, confidence

    def forward(self, student_rgb_patches, student_ir_patches,
                teacher_rgb_patches, teacher_ir_patches,
                grid_size):
        if (student_rgb_patches is None or student_ir_patches is None
                or teacher_rgb_patches is None or teacher_ir_patches is None):
            reference = next(
                (x for x in (student_rgb_patches, student_ir_patches,
                             teacher_rgb_patches, teacher_ir_patches)
                 if x is not None),
                None,
            )
            if reference is None:
                return torch.tensor(0.0)
            return self._zero_like(reference)

        if student_rgb_patches.numel() == 0 or student_ir_patches.numel() == 0:
            return self._zero_like(student_rgb_patches)

        if student_rgb_patches.shape != student_ir_patches.shape:
            raise ValueError(
                f"student_rgb_patches.shape={student_rgb_patches.shape} "
                f"!= student_ir_patches.shape={student_ir_patches.shape}"
            )
        if teacher_rgb_patches.shape != student_rgb_patches.shape:
            raise ValueError(
                f"teacher_rgb_patches.shape={teacher_rgb_patches.shape} "
                f"!= student_rgb_patches.shape={student_rgb_patches.shape}"
            )
        if teacher_ir_patches.shape != student_ir_patches.shape:
            raise ValueError(
                f"teacher_ir_patches.shape={teacher_ir_patches.shape} "
                f"!= student_ir_patches.shape={student_ir_patches.shape}"
            )

        B, N, _ = student_rgb_patches.shape
        if grid_size * grid_size != N:
            raise ValueError(
                f"grid_size²={grid_size**2} != N={N} for cross-modal patch alignment."
            )

        positions = self._build_patch_positions(
            grid_size, student_rgb_patches.device, student_rgb_patches.dtype)
        spatial_dist_sq = ((positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2).sum(-1)
        sigma = student_rgb_patches.new_full((B,), float(self.spatial_sigma))
        identity_prior = self._identity_prior(spatial_dist_sq, sigma)

        s_rgb = F.normalize(student_rgb_patches, dim=-1)
        s_ir = F.normalize(student_ir_patches, dim=-1)
        with torch.no_grad():
            t_rgb = F.normalize(teacher_rgb_patches, dim=-1)
            t_ir = F.normalize(teacher_ir_patches, dim=-1)
            teacher_target_rgb, confidence_rgb = self._teacher_targets(
                t_rgb, t_ir, positions, identity_prior, sigma)
            teacher_target_ir, confidence_ir = self._teacher_targets(
                t_ir, t_rgb, positions, identity_prior, sigma)

        student_rgb_log_prob = F.log_softmax(
            (torch.bmm(s_rgb, s_ir.transpose(1, 2)) / self.patch_temperature).clamp(-100, 100),
            dim=-1,
        )
        student_ir_log_prob = F.log_softmax(
            (torch.bmm(s_ir, s_rgb.transpose(1, 2)) / self.patch_temperature).clamp(-100, 100),
            dim=-1,
        )

        loss_rgb = -(teacher_target_rgb * student_rgb_log_prob).sum(dim=-1)
        loss_ir = -(teacher_target_ir * student_ir_log_prob).sum(dim=-1)
        total = (
            (loss_rgb * confidence_rgb).sum()
            + (loss_ir * confidence_ir).sum()
        )
        normalizer = confidence_rgb.sum() + confidence_ir.sum()
        return total / normalizer.clamp(min=1e-6)


class CrossModalMGCLLoss(nn.Module):
    """3-level cross-modal alignment: token + object + image.

    Distills single-modality student MG features toward teacher's
    RGB+IR fused MG features using softmax distillation with centering.
    Centers come from the shared MGCLLoss (passed in at forward time).
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def _distill_loss(self, student_feat, teacher_feat, center):
        """Softmax distillation: student log_softmax vs teacher softmax with center."""
        if student_feat.dim() == 3:
            bsz, n, dim = student_feat.shape
            student_feat = student_feat.reshape(bsz * n, dim)
            teacher_feat = teacher_feat.reshape(bsz * n, dim)

        student_feat = F.normalize(student_feat, dim=-1, eps=1e-6)
        teacher_feat = F.normalize(teacher_feat, dim=-1, eps=1e-6)

        s = F.log_softmax(student_feat / self.temperature, dim=-1)
        with torch.no_grad():
            t = F.softmax((teacher_feat - center) / (self.temperature * 0.4), dim=-1)
        return -torch.sum(t * s, dim=-1).mean()

    def forward(self, student_rgb_mg, student_ir_mg, teacher_fused_mg,
                center_token, center_object, center_image):
        """
        Args:
            student_rgb_mg: dict with 'token', 'object', 'image' from RGB-only student
            student_ir_mg:  dict with 'token', 'object', 'image' from IR-only student
            teacher_fused_mg: dict with 'token', 'object', 'image' from RGB+IR teacher
            center_token/object/image: centering buffers from MGCLLoss
        Returns:
            dict with token_loss, object_loss, image_loss, total
        """
        # Token-level
        token_rgb = self._distill_loss(
            student_rgb_mg['token'], teacher_fused_mg['token'], center_token)
        token_ir = self._distill_loss(
            student_ir_mg['token'], teacher_fused_mg['token'], center_token)
        token_loss = (token_rgb + token_ir) / 2.0

        # Object-level
        obj_rgb = self._distill_loss(
            student_rgb_mg['object'], teacher_fused_mg['object'], center_object)
        obj_ir = self._distill_loss(
            student_ir_mg['object'], teacher_fused_mg['object'], center_object)
        object_loss = (obj_rgb + obj_ir) / 2.0

        # Image-level
        img_rgb = self._distill_loss(
            student_rgb_mg['image'], teacher_fused_mg['image'], center_image)
        img_ir = self._distill_loss(
            student_ir_mg['image'], teacher_fused_mg['image'], center_image)
        image_loss = (img_rgb + img_ir) / 2.0

        total = (token_loss + object_loss + image_loss) / 3.0
        return {
            'token_loss': token_loss,
            'object_loss': object_loss,
            'image_loss': image_loss,
            'total': total,
        }


class ViewInvarianceLoss(nn.Module):
    """Small-angle affine view invariance loss (ATCL-style)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_original, feat_transformed):
        if feat_transformed is None:
            return torch.tensor(0.0, device=feat_original.device)

        feat_original = F.normalize(feat_original, dim=-1)
        feat_transformed = F.normalize(feat_transformed, dim=-1)
        bsz = feat_original.shape[0]

        sim = feat_original @ feat_transformed.t() / self.temperature
        labels = torch.arange(bsz, device=sim.device)
        loss_o2t = F.cross_entropy(sim, labels)
        loss_t2o = F.cross_entropy(sim.t(), labels)
        return (loss_o2t + loss_t2o) / 2


class CrossModalPairLoss(nn.Module):
    """Explicit RGB-only <-> IR-only alignment on paired samples."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, rgb_feat, ir_feat):
        if rgb_feat is None or ir_feat is None or rgb_feat.numel() == 0 or ir_feat.numel() == 0:
            device = None
            if rgb_feat is not None:
                device = rgb_feat.device
            elif ir_feat is not None:
                device = ir_feat.device
            device = device or torch.device("cpu")
            return torch.tensor(0.0, device=device)
        if rgb_feat.shape[0] != ir_feat.shape[0]:
            raise ValueError(
                f"rgb_feat.shape[0]={rgb_feat.shape[0]} != ir_feat.shape[0]={ir_feat.shape[0]}"
            )

        rgb_z = F.normalize(rgb_feat, dim=-1)
        ir_z = F.normalize(ir_feat, dim=-1)
        logits = torch.matmul(rgb_z, ir_z.t()) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_rgb = F.cross_entropy(logits, labels)
        loss_ir = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_rgb + loss_ir)


class CrossModalPairLossWithQueue(nn.Module):
    """RGB <-> IR alignment with momentum queue, hard negative mining, and
    curriculum temperature scheduling.

    Improvements over CrossModalPairLoss:
    - Momentum queue: stores recent embeddings to increase effective negatives
    - Hard negative mining: upweights top-K hardest negatives in logits
    - Curriculum temperature: cosine decay from temp_start to temp_end
    """

    def __init__(self, embed_dim=384, temperature=0.07,
                 queue_size=4096, hard_neg_topk=128, hard_neg_weight=2.0,
                 temp_start=0.2, temp_end=0.05, temp_warmup_epochs=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_temperature = temperature
        self.queue_size = queue_size
        self.hard_neg_topk = hard_neg_topk
        self.hard_neg_weight = hard_neg_weight
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_warmup_epochs = temp_warmup_epochs

        # Momentum queues
        self.register_buffer('rgb_queue', torch.randn(queue_size, embed_dim))
        self.register_buffer('ir_queue', torch.randn(queue_size, embed_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        # L2 normalize initial queue
        self.rgb_queue = F.normalize(self.rgb_queue, dim=1)
        self.ir_queue = F.normalize(self.ir_queue, dim=1)

    def get_temperature(self, epoch, total_epochs=40):
        """Cosine decay from temp_start to temp_end over temp_warmup_epochs."""
        if epoch >= self.temp_warmup_epochs:
            return self.temp_end
        progress = epoch / max(1, self.temp_warmup_epochs)
        # Cosine decay
        return self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (
            1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, rgb_feat, ir_feat):
        """Enqueue current batch features into the queue, with DDP gather."""
        # Gather across all GPUs if distributed
        if dist.is_available() and dist.is_initialized():
            rgb_gather = [torch.zeros_like(rgb_feat) for _ in range(dist.get_world_size())]
            ir_gather = [torch.zeros_like(ir_feat) for _ in range(dist.get_world_size())]
            dist.all_gather(rgb_gather, rgb_feat)
            dist.all_gather(ir_gather, ir_feat)
            rgb_feat = torch.cat(rgb_gather, dim=0)
            ir_feat = torch.cat(ir_gather, dim=0)

        batch_size = rgb_feat.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.rgb_queue[ptr:ptr + batch_size] = rgb_feat
            self.ir_queue[ptr:ptr + batch_size] = ir_feat
        else:
            # Wraparound
            remaining = self.queue_size - ptr
            self.rgb_queue[ptr:] = rgb_feat[:remaining]
            self.ir_queue[ptr:] = ir_feat[:remaining]
            overflow = batch_size - remaining
            if overflow > 0:
                self.rgb_queue[:overflow] = rgb_feat[remaining:remaining + overflow]
                self.ir_queue[:overflow] = ir_feat[remaining:remaining + overflow]

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def _hard_negative_logits(self, logits, labels, topk):
        """Upweight top-K hardest negatives in the logit matrix."""
        B = logits.shape[0]
        # Create negative mask (1 for negatives, 0 for positives)
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask[torch.arange(B, device=logits.device), labels] = False

        # Get logits for negatives only, find top-K hardest
        neg_logits = logits.clone()
        neg_logits[~neg_mask] = float('-inf')

        k = min(topk, neg_mask.sum(dim=1).min().item())
        if k <= 0:
            return logits

        _, hard_indices = neg_logits.topk(k, dim=1)  # [B, K]

        # Create weight matrix: 1.0 for all, hard_neg_weight for hard negatives
        weights = torch.ones_like(logits)
        weights.scatter_(1, hard_indices, self.hard_neg_weight)
        # Don't weight the positive
        weights[torch.arange(B, device=logits.device), labels] = 1.0

        return logits * weights

    def forward(self, rgb_feat, ir_feat, epoch=0, total_epochs=40):
        if rgb_feat is None or ir_feat is None or rgb_feat.numel() == 0 or ir_feat.numel() == 0:
            device = None
            if rgb_feat is not None:
                device = rgb_feat.device
            elif ir_feat is not None:
                device = ir_feat.device
            device = device or torch.device("cpu")
            return torch.tensor(0.0, device=device)

        if rgb_feat.shape[0] != ir_feat.shape[0]:
            raise ValueError(
                f"rgb_feat.shape[0]={rgb_feat.shape[0]} != ir_feat.shape[0]={ir_feat.shape[0]}")

        B = rgb_feat.shape[0]
        temperature = self.get_temperature(epoch, total_epochs)

        # L2 normalize
        rgb_z = F.normalize(rgb_feat.float(), dim=-1)
        ir_z = F.normalize(ir_feat.float(), dim=-1)

        # Ensure queue dtype matches
        rgb_queue = self.rgb_queue.clone().to(dtype=rgb_z.dtype)
        ir_queue = self.ir_queue.clone().to(dtype=ir_z.dtype)

        # Compute logits: batch pairs + queue negatives
        # RGB→IR: query=rgb, keys=ir_batch + ir_queue
        ir_all = torch.cat([ir_z, ir_queue], dim=0)  # [B+Q, D]
        logits_rgb2ir = torch.matmul(rgb_z, ir_all.t()) / temperature  # [B, B+Q]

        # IR→RGB: query=ir, keys=rgb_batch + rgb_queue
        rgb_all = torch.cat([rgb_z, rgb_queue], dim=0)  # [B+Q, D]
        logits_ir2rgb = torch.matmul(ir_z, rgb_all.t()) / temperature  # [B, B+Q]

        # Labels: diagonal (matching pairs are at index i)
        labels = torch.arange(B, device=rgb_z.device)

        # Hard negative mining
        logits_rgb2ir = self._hard_negative_logits(logits_rgb2ir, labels, self.hard_neg_topk)
        logits_ir2rgb = self._hard_negative_logits(logits_ir2rgb, labels, self.hard_neg_topk)

        loss_rgb = F.cross_entropy(logits_rgb2ir, labels)
        loss_ir = F.cross_entropy(logits_ir2rgb, labels)

        # Update queue
        self._dequeue_and_enqueue(rgb_z.detach(), ir_z.detach())

        return 0.5 * (loss_rgb + loss_ir)


class GrayscaleBridgeLoss(nn.Module):
    """Bridge loss: RGB→Gray + Gray→IR alignment.

    Uses two CrossModalPairLossWithQueue instances to create a bridging path:
    RGB ↔ Grayscale ↔ IR, where grayscale shares visual structure with RGB
    and channel dimensionality with IR.
    """

    def __init__(self, embed_dim=384, **kwargs):
        super().__init__()
        self.rgb_gray_loss = CrossModalPairLossWithQueue(embed_dim=embed_dim, **kwargs)
        self.gray_ir_loss = CrossModalPairLossWithQueue(embed_dim=embed_dim, **kwargs)

    def forward(self, rgb_feat, gray_feat, ir_feat, epoch=0, total_epochs=40):
        loss_rgb_gray = self.rgb_gray_loss(rgb_feat, gray_feat, epoch, total_epochs)
        loss_gray_ir = self.gray_ir_loss(gray_feat, ir_feat, epoch, total_epochs)
        return 0.5 * (loss_rgb_gray + loss_gray_ir)


class ViewBridgeLoss(nn.Module):
    """
    View-domain bridge loss with Sinkhorn-Knopp soft targets.

    Student predicts soft equipartitioned assignments from teacher.
    Loss = Cross-entropy between student logits and teacher soft targets,
    averaged over all views.
    """

    def __init__(self, temperature=0.1, entropy_reg=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.entropy_reg = entropy_reg  # entropy regularization to prevent collapse

    @staticmethod
    def target_entropy(teacher_targets_list):
        """Average entropy of the soft bridge targets."""
        if teacher_targets_list is None or len(teacher_targets_list) == 0:
            return None
        eps = 1e-8
        entropy = 0.0
        n = 0
        for target in teacher_targets_list:
            target = target.clamp(min=eps)
            entropy = entropy + (-(target * target.log()).sum(dim=-1).mean())
            n += 1
        return entropy / max(1, n)

    def forward(self, student_logits_list, teacher_targets_list=None):
        """
        Args:
            student_logits_list: list of [B, K] student similarity logits
            teacher_targets_list: list of [B, K] soft assignment targets from
                                  Sinkhorn-Knopp (teacher side, no grad).
                                  If None, falls back to consistency loss.

        Returns:
            loss: scalar
        """
        if student_logits_list is None or len(student_logits_list) == 0:
            device = torch.device("cpu")
            return torch.tensor(0.0, device=device)

        if teacher_targets_list is not None and len(teacher_targets_list) > 0:
            # Cross-entropy with soft Sinkhorn targets
            loss = 0.0
            n = 0
            for s_logits, t_target in zip(student_logits_list, teacher_targets_list):
                # Student log-probabilities
                log_p = F.log_softmax(s_logits / self.temperature, dim=-1)
                p = log_p.exp()
                # Soft cross-entropy: -sum_k Q_k * log(p_k)
                ce = (-t_target * log_p).sum(dim=-1).mean()
                # Entropy regularization: encourage student to maintain diversity
                # Prevents collapse where student becomes too confident
                student_entropy = -(p * log_p).sum(dim=-1).mean()
                loss = loss + ce - self.entropy_reg * student_entropy
                n += 1
            return loss / max(1, n)
        else:
            # Fallback: pairwise consistency (old behavior)
            eps = 1e-8
            probs = [F.softmax(logits / self.temperature, dim=-1)
                     for logits in student_logits_list]

            consistency = 0.0
            n_pairs = 0
            for i in range(len(probs)):
                for j in range(i + 1, len(probs)):
                    pi = probs[i].clamp(min=eps)
                    pj = probs[j].clamp(min=eps)
                    kl_ij = torch.sum(pi * (pi.log() - pj.log()), dim=-1).mean()
                    kl_ji = torch.sum(pj * (pj.log() - pi.log()), dim=-1).mean()
                    consistency = consistency + 0.5 * (kl_ij + kl_ji)
                    n_pairs += 1
            return consistency / max(1, n_pairs)


class UncertaintyWeighting(nn.Module):
    """Kendall Uncertainty Weighting for multi-task loss balancing.

    Each auxiliary loss gets a learnable log-variance parameter s_i.
    Weighted loss = exp(-s_i) * L_i + 0.5 * s_i
    s_i clamped to [-6, 6] for FP16 safety (exp range ~[0.0025, 403]).
    """

    def __init__(self, loss_names, init_weights):
        """
        Args:
            loss_names: list of auxiliary loss names
            init_weights: dict {name: w_i}, s_i initialized to -log(w_i)
        """
        super().__init__()
        self.loss_names = list(loss_names)
        self.log_vars = nn.ParameterDict()
        for name in self.loss_names:
            w = init_weights.get(name, 1.0)
            init_val = -math.log(max(w, 1e-8))
            self.log_vars[name] = nn.Parameter(torch.tensor(init_val))

    def forward(self, raw_losses, loss_scales=None):
        """
        Args:
            raw_losses: dict {name: scalar tensor} for active auxiliary losses

        Returns:
            weighted_sum: scalar total auxiliary loss
            effective_weights: dict {ew_<name>: float} current effective weights
        """
        weighted_sum = 0.0
        effective_weights = {}
        for name in self.loss_names:
            s = self.log_vars[name].clamp(-6.0, 6.0)
            ew = torch.exp(-s)
            scale = 1.0 if loss_scales is None else float(loss_scales.get(name, 1.0))
            if name in raw_losses:
                weighted_sum = weighted_sum + scale * (ew * raw_losses[name] + 0.5 * s)
            else:
                # Loss absent this batch — still touch the param so grad is
                # never None (prevents NCCL all_reduce deadlock across ranks)
                weighted_sum = weighted_sum + 0.0 * s
            effective_weights[f'ew_{name}'] = scale * ew.item()
        return weighted_sum, effective_weights


class PretrainingLoss(nn.Module):
    """Combined pretraining loss.

    Active losses: DINO (anchor, w=1), MGCL, View, Bridge, TCL.
    Video losses: weighted DINO_video, MGCL_video, Bridge_video, TCL_patch.
    """

    def __init__(self, dino_loss, mgcl_loss, view_loss,
                 cross_modal_pair_loss=None,
                 cross_modal_patch_loss=None,
                 cross_modal_mgcl_loss=None,
                 view_bridge_loss=None, tcl_loss=None,
                 gray_bridge_loss=None,
                 w_dino_video=0.2, w_mgcl=1.0, w_view=0.5, w_bridge=0.5,
                 w_tcl=0.1, w_tcl_patch=0.5, w_align_rgbir=0.0,
                 w_align_rgbir_patch=0.0, w_align_mgcl=0.0,
                 w_gray_bridge=0.0,
                 align_ramp_epochs=0, total_epochs=40,
                 adaptive_weighting=False):
        super().__init__()
        self.dino_loss = dino_loss
        self.mgcl_loss = mgcl_loss
        self.view_loss = view_loss
        self.cross_modal_pair_loss = cross_modal_pair_loss
        self.cross_modal_patch_loss = cross_modal_patch_loss
        self.cross_modal_mgcl_loss = cross_modal_mgcl_loss
        self.view_bridge_loss = view_bridge_loss
        self.tcl_loss = tcl_loss
        self.gray_bridge_loss = gray_bridge_loss

        self.w_dino_video = w_dino_video
        self.w_mgcl = w_mgcl
        self.w_view = w_view
        self.w_bridge = w_bridge
        self.w_tcl = w_tcl
        self.w_tcl_patch = w_tcl_patch
        self.w_align_rgbir = w_align_rgbir
        self.w_align_rgbir_patch = w_align_rgbir_patch
        self.w_align_mgcl = w_align_mgcl
        self.w_gray_bridge = w_gray_bridge
        self.align_ramp_epochs = align_ramp_epochs
        self.total_epochs = total_epochs

        self.adaptive_weighting = adaptive_weighting

        if adaptive_weighting:
            aux_names = ['dino_video', 'mgcl', 'view', 'bridge']
            init_weights = {
                'dino_video': w_dino_video,
                'mgcl': w_mgcl,
                'view': w_view, 'bridge': w_bridge,
            }
            if cross_modal_pair_loss is not None:
                aux_names.append('align_rgbir')
                init_weights['align_rgbir'] = w_align_rgbir
            if cross_modal_patch_loss is not None:
                aux_names.append('align_rgbir_patch')
                init_weights['align_rgbir_patch'] = w_align_rgbir_patch
            if cross_modal_mgcl_loss is not None:
                aux_names.append('align_mgcl')
                init_weights['align_mgcl'] = w_align_mgcl
            if gray_bridge_loss is not None:
                aux_names.append('gray_bridge')
                init_weights['gray_bridge'] = w_gray_bridge
            if tcl_loss is not None:
                aux_names.extend(['tcl', 'tcl_patch',
                                  'mgcl_video', 'bridge_video'])
                init_weights.update({
                    'tcl': w_tcl, 'tcl_patch': w_tcl_patch,
                    'mgcl_video': w_mgcl,
                    'bridge_video': w_bridge,
                })
            self.uncertainty = UncertaintyWeighting(aux_names, init_weights)

    @staticmethod
    def _aux_scales(aux_weight_scale=1.0, video_aux_weight_scale=1.0):
        scales = {
            'mgcl': aux_weight_scale,
            'view': aux_weight_scale,
            'bridge': aux_weight_scale,
            'align_rgbir': aux_weight_scale,
            'align_rgbir_patch': aux_weight_scale,
            'align_mgcl': aux_weight_scale,
            'gray_bridge': aux_weight_scale,
            'dino_video': video_aux_weight_scale,
            'tcl': video_aux_weight_scale,
            'tcl_patch': video_aux_weight_scale,
            'mgcl_video': video_aux_weight_scale,
            'bridge_video': video_aux_weight_scale,
        }
        return scales

    def forward(self, student_out, teacher_out, epoch,
                student_mg=None, teacher_mg=None,
                feat_original=None, feat_view=None,
                align_rgb_feat=None, align_ir_feat=None,
                align_gray_feat=None,
                align_rgb_patches=None, align_ir_patches=None,
                align_teacher_rgb_patches=None, align_teacher_ir_patches=None,
                align_patch_grid_size=None,
                align_rgb_mg=None, align_ir_mg=None,
                align_teacher_fused_mg=None,
                bridge_logits_list=None,
                tcl_features=None, num_frames=1,
                aux_weight_scale=1.0,
                video_aux_weight_scale=1.0,
                # Video integration parameters
                video_student_out=None, video_teacher_out=None,
                video_ncrops=None,
                video_teacher_indices=None,
                video_student_patches=None, video_teacher_patches=None,
                video_timestamps=None,
                video_student_mg=None, video_teacher_mg=None,
                video_bridge_logits_list=None, grid_size=16):
        loss_dict = {}
        individual_losses = {}
        teacher_outputs_for_center = [teacher_out]

        # DINO loss — always weight=1.0, anchor
        l_dino = self.dino_loss(
            student_out, teacher_out, epoch, update_center=False)
        loss_dict['dino'] = l_dino.item()
        individual_losses['dino'] = l_dino
        total = l_dino

        # --- Compute raw auxiliary losses ---
        aux_losses = {}

        if student_mg is not None and teacher_mg is not None:
            mgcl_out = self.mgcl_loss(student_mg, teacher_mg, update_center=False)
            l_mgcl = mgcl_out['total']
            loss_dict['mgcl'] = l_mgcl.item()
            loss_dict['mgcl_token'] = mgcl_out['token_loss'].item()
            loss_dict['mgcl_object'] = mgcl_out['object_loss'].item()
            loss_dict['mgcl_image'] = mgcl_out['image_loss'].item()
            individual_losses['mgcl'] = l_mgcl
            aux_losses['mgcl'] = l_mgcl

        if feat_original is not None and feat_view is not None:
            l_view = self.view_loss(feat_original, feat_view)
            loss_dict['view'] = l_view.item()
            individual_losses['view'] = l_view
            aux_losses['view'] = l_view

        if self.cross_modal_pair_loss is not None and align_rgb_feat is not None and align_ir_feat is not None:
            # Pass epoch/total_epochs if the loss supports it (CrossModalPairLossWithQueue)
            if isinstance(self.cross_modal_pair_loss, CrossModalPairLossWithQueue):
                l_align = self.cross_modal_pair_loss(align_rgb_feat, align_ir_feat,
                                                     epoch=epoch, total_epochs=self.total_epochs)
            else:
                l_align = self.cross_modal_pair_loss(align_rgb_feat, align_ir_feat)
            loss_dict['align_rgbir'] = l_align.item()
            individual_losses['align_rgbir'] = l_align
            aux_losses['align_rgbir'] = l_align
        else:
            loss_dict['align_rgbir'] = 0.0

        if (self.cross_modal_patch_loss is not None
                and align_rgb_patches is not None
                and align_ir_patches is not None
                and align_teacher_rgb_patches is not None
                and align_teacher_ir_patches is not None
                and align_patch_grid_size is not None):
            l_align_patch = self.cross_modal_patch_loss(
                align_rgb_patches, align_ir_patches,
                align_teacher_rgb_patches, align_teacher_ir_patches,
                align_patch_grid_size,
            )
            loss_dict['align_rgbir_patch'] = l_align_patch.item()
            individual_losses['align_rgbir_patch'] = l_align_patch
            aux_losses['align_rgbir_patch'] = l_align_patch
        else:
            loss_dict['align_rgbir_patch'] = 0.0

        # Grayscale bridge loss
        if (self.gray_bridge_loss is not None
                and align_rgb_feat is not None and align_ir_feat is not None
                and align_gray_feat is not None):
            l_gray_bridge = self.gray_bridge_loss(
                align_rgb_feat, align_gray_feat, align_ir_feat,
                epoch=epoch, total_epochs=self.total_epochs)
            loss_dict['gray_bridge'] = l_gray_bridge.item()
            individual_losses['gray_bridge'] = l_gray_bridge
            aux_losses['gray_bridge'] = l_gray_bridge
        else:
            loss_dict['gray_bridge'] = 0.0

        # 3-level cross-modal MGCL alignment
        if (self.cross_modal_mgcl_loss is not None
                and align_rgb_mg is not None
                and align_ir_mg is not None
                and align_teacher_fused_mg is not None):
            cm_mgcl_out = self.cross_modal_mgcl_loss(
                align_rgb_mg, align_ir_mg, align_teacher_fused_mg,
                self.mgcl_loss.center_token,
                self.mgcl_loss.center_object,
                self.mgcl_loss.center_image,
            )
            l_align_mgcl = cm_mgcl_out['total']
            loss_dict['align_mgcl'] = l_align_mgcl.item()
            loss_dict['align_mgcl_token'] = cm_mgcl_out['token_loss'].item()
            loss_dict['align_mgcl_object'] = cm_mgcl_out['object_loss'].item()
            loss_dict['align_mgcl_image'] = cm_mgcl_out['image_loss'].item()
            individual_losses['align_mgcl'] = l_align_mgcl
            aux_losses['align_mgcl'] = l_align_mgcl
        else:
            loss_dict['align_mgcl'] = 0.0

        if self.view_bridge_loss is not None and bridge_logits_list is not None:
            if isinstance(bridge_logits_list, tuple) and len(bridge_logits_list) == 2:
                student_logits, teacher_targets = bridge_logits_list
                l_bridge = self.view_bridge_loss(student_logits, teacher_targets)
                bridge_entropy = self.view_bridge_loss.target_entropy(teacher_targets)
            else:
                l_bridge = self.view_bridge_loss(bridge_logits_list)
                bridge_entropy = None
            loss_dict['bridge'] = l_bridge.item()
            if bridge_entropy is not None:
                loss_dict['bridge_entropy'] = bridge_entropy.item()
                loss_dict['bridge_kl'] = max(0.0, (l_bridge - bridge_entropy).item())
            else:
                loss_dict['bridge_entropy'] = 0.0
                loss_dict['bridge_kl'] = 0.0
            individual_losses['bridge'] = l_bridge
            aux_losses['bridge'] = l_bridge
        else:
            loss_dict['bridge'] = 0.0
            loss_dict['bridge_entropy'] = 0.0
            loss_dict['bridge_kl'] = 0.0

        # Global TCL (InfoNCE)
        if self.tcl_loss is not None and tcl_features is not None:
            l_tcl = self.tcl_loss(tcl_features, num_frames, timestamps=video_timestamps)
            loss_dict['tcl'] = l_tcl.item()
            if getattr(self.tcl_loss, 'loss_name', 'tcl') != 'tcl':
                loss_dict[self.tcl_loss.loss_name] = l_tcl.item()
            individual_losses['tcl'] = l_tcl
            aux_losses['tcl'] = l_tcl
        else:
            loss_dict['tcl'] = 0.0

        # --- Video losses ---

        # Video DINO
        if video_student_out is not None and video_teacher_out is not None:
            l_dino_video = self.dino_loss(
                video_student_out, video_teacher_out, epoch,
                ncrops=video_ncrops,
                teacher_indices=video_teacher_indices,
                update_center=False)
            loss_dict['dino_video'] = l_dino_video.item()
            individual_losses['dino_video'] = l_dino_video
            aux_losses['dino_video'] = l_dino_video
            teacher_outputs_for_center.append(video_teacher_out)
        else:
            loss_dict['dino_video'] = 0.0

        # Patch-level TCL
        if (self.tcl_loss is not None
                and video_student_patches is not None
                and video_teacher_patches is not None):
            l_tcl_patch = self.tcl_loss.forward_patch(
                video_student_patches, video_teacher_patches,
                num_frames, grid_size, timestamps=video_timestamps)
            loss_dict['tcl_patch'] = l_tcl_patch.item()
            if getattr(self.tcl_loss, 'patch_loss_name', 'tcl_patch') != 'tcl_patch':
                loss_dict[self.tcl_loss.patch_loss_name] = l_tcl_patch.item()
            individual_losses['tcl_patch'] = l_tcl_patch
            aux_losses['tcl_patch'] = l_tcl_patch
        else:
            loss_dict['tcl_patch'] = 0.0

        # Video MGCL
        if video_student_mg is not None and video_teacher_mg is not None:
            v_mgcl_out = self.mgcl_loss(video_student_mg, video_teacher_mg, update_center=False)
            l_mgcl_video = v_mgcl_out['total']
            loss_dict['mgcl_video'] = l_mgcl_video.item()
            individual_losses['mgcl_video'] = l_mgcl_video
            aux_losses['mgcl_video'] = l_mgcl_video
        else:
            loss_dict['mgcl_video'] = 0.0

        # Update MGCL centers once per step so image/video branches use the
        # same teacher centering rule without order-dependent double updates.
        mg_teacher_outputs = []
        if teacher_mg is not None:
            mg_teacher_outputs.append(teacher_mg)
        if video_teacher_mg is not None:
            mg_teacher_outputs.append(video_teacher_mg)
        if mg_teacher_outputs:
            self.mgcl_loss.update_centers_multi(mg_teacher_outputs)

        # Video Bridge
        if (self.view_bridge_loss is not None
                and video_bridge_logits_list is not None):
            if isinstance(video_bridge_logits_list, tuple) and len(video_bridge_logits_list) == 2:
                v_s_logits, v_t_targets = video_bridge_logits_list
                l_bridge_video = self.view_bridge_loss(v_s_logits, v_t_targets)
                bridge_video_entropy = self.view_bridge_loss.target_entropy(v_t_targets)
            else:
                l_bridge_video = self.view_bridge_loss(video_bridge_logits_list)
                bridge_video_entropy = None
            loss_dict['bridge_video'] = l_bridge_video.item()
            if bridge_video_entropy is not None:
                loss_dict['bridge_video_entropy'] = bridge_video_entropy.item()
                loss_dict['bridge_video_kl'] = max(
                    0.0, (l_bridge_video - bridge_video_entropy).item())
            else:
                loss_dict['bridge_video_entropy'] = 0.0
                loss_dict['bridge_video_kl'] = 0.0
            individual_losses['bridge_video'] = l_bridge_video
            aux_losses['bridge_video'] = l_bridge_video
        else:
            loss_dict['bridge_video'] = 0.0
            loss_dict['bridge_video_entropy'] = 0.0
            loss_dict['bridge_video_kl'] = 0.0

        # Update DINO center once per step so image/video branches share a
        # consistent teacher distribution without double-counting momentum.
        if teacher_outputs_for_center:
            self.dino_loss.update_center(torch.cat(teacher_outputs_for_center, dim=0))

        # --- Weight auxiliary losses ---
        aux_scales = self._aux_scales(
            aux_weight_scale=aux_weight_scale,
            video_aux_weight_scale=video_aux_weight_scale,
        )
        if self.adaptive_weighting:
            aux_total, ew = self.uncertainty(aux_losses, loss_scales=aux_scales)
            total = total + aux_total
            loss_dict.update(ew)
        else:
            # Static weights
            # Alignment ramp: gradually increase align weights over align_ramp_epochs
            align_ramp = 1.0
            if self.align_ramp_epochs > 0:
                align_ramp = min(1.0, (epoch + 1) / self.align_ramp_epochs)
            weight_map = {
                'dino_video': self.w_dino_video,
                'mgcl': self.w_mgcl, 'view': self.w_view,
                'bridge': self.w_bridge,
                'align_rgbir': self.w_align_rgbir * align_ramp,
                'align_rgbir_patch': self.w_align_rgbir_patch * align_ramp,
                'align_mgcl': self.w_align_mgcl * align_ramp,
                'gray_bridge': self.w_gray_bridge * align_ramp,
                'tcl': self.w_tcl,
                'tcl_patch': self.w_tcl_patch,
                'mgcl_video': self.w_mgcl, 'bridge_video': self.w_bridge,
            }
            for key, weight in weight_map.items():
                if key in aux_losses:
                    total = total + aux_scales.get(key, 1.0) * weight * aux_losses[key]

        loss_dict['total'] = total.item()
        return total, loss_dict, individual_losses

    def get_pcgrad_aux_terms(self, individual_losses,
                             aux_weight_scale=1.0,
                             video_aux_weight_scale=1.0):
        """Build the exact auxiliary objectives used by backward().

        Returns:
            dict {name: (weight, loss_tensor)} compatible with pcgrad_backward.
            When adaptive weighting is enabled, the returned loss tensor already
            includes both exp(-s) * L and the 0.5 * s regularizer, so the
            external weight is fixed to 1.0.
        """
        aux_names = (
            'dino_video', 'mgcl', 'view', 'bridge', 'align_rgbir', 'align_rgbir_patch',
            'align_mgcl', 'gray_bridge', 'tcl', 'tcl_patch',
            'mgcl_video', 'bridge_video',
        )
        aux_terms = {}
        aux_scales = self._aux_scales(
            aux_weight_scale=aux_weight_scale,
            video_aux_weight_scale=video_aux_weight_scale,
        )
        for key in aux_names:
            if key not in individual_losses:
                continue
            scale = aux_scales.get(key, 1.0)
            if self.adaptive_weighting:
                if not hasattr(self, 'uncertainty') or key not in self.uncertainty.log_vars:
                    continue
                s = self.uncertainty.log_vars[key].clamp(-6.0, 6.0)
                aux_terms[key] = (1.0, scale * (torch.exp(-s) * individual_losses[key] + 0.5 * s))
            else:
                weight = getattr(self, f'w_{key}', None)
                if weight is None:
                    base_key = key.replace('_video', '')
                    weight = getattr(self, f'w_{base_key}', 1.0)
                aux_terms[key] = (scale * weight, individual_losses[key])
        return aux_terms
