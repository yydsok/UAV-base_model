"""
Multi-modal DINO pretraining loss functions.

Contains:
- DINOLoss: Standard DINO self-distillation loss
- MGCLLoss: Multi-granularity contrastive loss (token/object/image)
- CrossModalAlignLoss: RGB-IR cross-modal alignment loss
- ViewInvarianceLoss: Affine viewpoint invariance loss (ATCL, small-angle)
- ViewBridgeLoss: View-domain bridge loss (unsupervised prototype consistency)
- PretrainingLoss: Combined weighted loss
"""

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

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
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

        s = F.log_softmax(student_feat / self.temperature, dim=-1)
        with torch.no_grad():
            t = F.softmax((teacher_feat - center) / (self.temperature * 0.4), dim=-1)
        return -torch.sum(t * s, dim=-1).mean()

    def forward(self, student_mg, teacher_mg):
        token_loss = self._contrastive_loss(student_mg['token'], teacher_mg['token'], self.center_token)
        object_loss = self._contrastive_loss(student_mg['object'], teacher_mg['object'], self.center_object)
        image_loss = self._contrastive_loss(student_mg['image'], teacher_mg['image'], self.center_image)
        total = token_loss + object_loss + image_loss

        with torch.no_grad():
            self._update_center(self.center_token, teacher_mg['token'])
            self._update_center(self.center_object, teacher_mg['object'])
            self._update_center(self.center_image, teacher_mg['image'])

        return {
            'token_loss': token_loss,
            'object_loss': object_loss,
            'image_loss': image_loss,
            'total': total,
        }

    @torch.no_grad()
    def _update_center(self, center, teacher_feat):
        if teacher_feat.dim() == 3:
            batch_center = teacher_feat.mean(dim=[0, 1], keepdim=False).unsqueeze(0)
        elif teacher_feat.dim() == 2:
            batch_center = teacher_feat.mean(dim=0, keepdim=True)
        else:
            return

        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        center.data = (center.data * self.center_momentum
                       + batch_center * (1 - self.center_momentum))


class CrossModalAlignLoss(nn.Module):
    """Cross-modal alignment loss with InfoNCE over patch positions."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_rgb, feat_ir, modality_mask):
        both_available = (modality_mask[:, 0] * modality_mask[:, 1]).bool()
        if both_available.sum() == 0:
            return torch.tensor(0.0, device=feat_rgb.device, dtype=feat_rgb.dtype)

        rgb = F.normalize(feat_rgb[both_available], dim=-1, eps=1e-8)
        ir = F.normalize(feat_ir[both_available], dim=-1, eps=1e-8)
        bsz, n_tokens, _ = rgb.shape

        sim_matrix = torch.bmm(rgb, ir.transpose(1, 2)) / self.temperature
        sim_matrix = sim_matrix.clamp(min=-100, max=100)
        labels = torch.arange(n_tokens, device=rgb.device).unsqueeze(0).expand(bsz, -1)

        loss_rgb2ir = F.cross_entropy(sim_matrix.reshape(bsz * n_tokens, n_tokens), labels.reshape(bsz * n_tokens))
        loss_ir2rgb = F.cross_entropy(sim_matrix.transpose(1, 2).reshape(bsz * n_tokens, n_tokens), labels.reshape(bsz * n_tokens))
        return (loss_rgb2ir + loss_ir2rgb) / 2


class TCLLoss(nn.Module):
    """Time-Contrastive Learning loss (InfoNCE).

    Multi-frame: all frame pairs within same sequence are positives,
    frames from different sequences are negatives.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, frame_features, num_frames):
        """
        Args:
            frame_features: [B*T, d] pooled features per frame
            num_frames: int T (>=2)
        Returns:
            scalar loss
        """
        if num_frames <= 1:
            return torch.tensor(0.0, device=frame_features.device,
                                dtype=frame_features.dtype)

        BT, d = frame_features.shape
        B = BT // num_frames
        T = num_frames

        # L2 normalize
        feats = F.normalize(frame_features, dim=-1, eps=1e-8)

        # Similarity matrix [BT, BT]
        sim = feats @ feats.t() / self.temperature
        sim = sim.clamp(min=-100, max=100)

        # Positive mask: same-sequence frame pairs (block-diagonal, exclude self)
        seq_ids = torch.arange(B, device=feats.device).unsqueeze(1).expand(B, T).reshape(BT)
        pos_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0))  # [BT, BT]
        self_mask = torch.eye(BT, device=feats.device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask  # exclude self

        # InfoNCE: for each anchor, log(sum_pos_exp / sum_all_exp)
        # Denominator: all except self
        neg_mask = ~self_mask
        log_denom = torch.logsumexp(sim.masked_fill(~neg_mask, -1e9), dim=1)  # [BT]

        # Numerator: log-sum-exp over positives for each anchor
        log_numer = torch.logsumexp(sim.masked_fill(~pos_mask, -1e9), dim=1)  # [BT]

        loss = -(log_numer - log_denom).mean()
        return loss


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


class ViewBridgeLoss(nn.Module):
    """
    Unsupervised large-view-difference bridge loss.

    Uses:
    1) Cross-view distribution consistency (symmetric KL)
    2) Per-sample sharpness (encourage confident prototype assignment)
    3) Batch-level balance (maximize prototype usage entropy)
    """

    def __init__(self, temperature=1.0, lambda_sharp=0.0, lambda_balance=0.02):
        super().__init__()
        self.temperature = temperature
        self.lambda_sharp = lambda_sharp
        self.lambda_balance = lambda_balance

    def forward(self, logits_list):
        if logits_list is None or len(logits_list) < 2:
            if logits_list is None:
                device = torch.device("cpu")
            else:
                device = logits_list[0].device
            return torch.tensor(0.0, device=device)

        eps = 1e-8
        probs = [F.softmax(logits / self.temperature, dim=-1) for logits in logits_list]

        # Pairwise symmetric KL consistency across views
        # With temperature=1.0 the distributions are soft enough for KL to produce
        # meaningful gradients (old temperature=0.1 caused near-one-hot → KL≈0).
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
        consistency = consistency / max(1, n_pairs)

        # Sharpness: encourage confident prototype assignment per sample
        # (minimize entropy = encourage sharp assignments)
        # NOTE: set lambda_sharp=0 when using high temperature to avoid conflict
        sharpness = 0.0
        if self.lambda_sharp > 0:
            for p in probs:
                sharpness = sharpness + (-torch.sum(p * p.clamp(min=eps).log(), dim=-1).mean())
            sharpness = sharpness / len(probs)

        # Balance: maximize entropy of batch-level prototype usage
        # (negative sign: we want to maximize this entropy, so minimize its negative)
        concat_p = torch.cat(probs, dim=0)
        mean_p = concat_p.mean(dim=0).clamp(min=eps)
        balance = -torch.sum(mean_p * mean_p.log())

        return consistency + self.lambda_sharp * sharpness - self.lambda_balance * balance


class PretrainingLoss(nn.Module):
    """Combined pretraining loss."""

    def __init__(self, dino_loss, mgcl_loss, align_loss, view_loss,
                 view_bridge_loss=None, tcl_loss=None,
                 w_mgcl=1.0, w_align=1.0, w_view=0.5, w_bridge=0.5,
                 w_latent=1.0, w_rec=1.0, w_tcl=0.1):
        super().__init__()
        self.dino_loss = dino_loss
        self.mgcl_loss = mgcl_loss
        self.align_loss = align_loss
        self.view_loss = view_loss
        self.view_bridge_loss = view_bridge_loss
        self.tcl_loss = tcl_loss

        self.w_mgcl = w_mgcl
        self.w_align = w_align
        self.w_view = w_view
        self.w_bridge = w_bridge
        self.w_latent = w_latent
        self.w_rec = w_rec
        self.w_tcl = w_tcl

    def forward(self, student_out, teacher_out, epoch,
                student_mg=None, teacher_mg=None,
                feat_rgb=None, feat_ir=None, modality_mask=None,
                feat_original=None, feat_view=None,
                bridge_logits_list=None,
                completion_losses=None,
                tcl_features=None, num_frames=1):
        loss_dict = {}

        l_dino = self.dino_loss(student_out, teacher_out, epoch)
        loss_dict['dino'] = l_dino.item()
        total = l_dino

        if student_mg is not None and teacher_mg is not None:
            mgcl_out = self.mgcl_loss(student_mg, teacher_mg)
            l_mgcl = mgcl_out['total']
            total = total + self.w_mgcl * l_mgcl
            loss_dict['mgcl'] = l_mgcl.item()
            loss_dict['mgcl_token'] = mgcl_out['token_loss'].item()
            loss_dict['mgcl_object'] = mgcl_out['object_loss'].item()
            loss_dict['mgcl_image'] = mgcl_out['image_loss'].item()

        if feat_rgb is not None and feat_ir is not None and modality_mask is not None:
            l_align = self.align_loss(feat_rgb, feat_ir, modality_mask)
            total = total + self.w_align * l_align
            loss_dict['align'] = l_align.item()

        if feat_original is not None and feat_view is not None:
            l_view = self.view_loss(feat_original, feat_view)
            total = total + self.w_view * l_view
            loss_dict['view'] = l_view.item()

        if self.view_bridge_loss is not None and bridge_logits_list is not None:
            l_bridge = self.view_bridge_loss(bridge_logits_list)
            total = total + self.w_bridge * l_bridge
            loss_dict['bridge'] = l_bridge.item()

        if completion_losses is not None:
            l_latent = completion_losses.get('latent_loss', torch.tensor(0.0, device=total.device))
            l_rec = completion_losses.get('rec_loss', torch.tensor(0.0, device=total.device))
            total = total + self.w_latent * l_latent + self.w_rec * l_rec
            loss_dict['latent'] = l_latent.item()
            loss_dict['rec'] = l_rec.item()

        # --- TCL: time-contrastive learning ---
        l_tcl = 0.0
        if self.tcl_loss is not None and tcl_features is not None:
            l_tcl = self.tcl_loss(tcl_features, num_frames)
        total = total + self.w_tcl * l_tcl
        loss_dict['tcl'] = l_tcl.item() if isinstance(l_tcl, torch.Tensor) else l_tcl

        loss_dict['total'] = total.item()
        return total, loss_dict
