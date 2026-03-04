"""
Masked Modality Completion module.
Handles missing modalities by learning to reconstruct features of absent
modalities from available ones.

Reference: SkySense++ Masked Modality Completion (Section 3.3)

Architecture:
- Per-modality encoder: maps features to unified latent space
- Codebook (optional VQ): vector quantization for robust latent representations
- Router: selects original vs reconstructed features based on modality availability
- Per-modality decoder: reconstructs features from latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Encode modality-specific features to unified latent space."""
    def __init__(self, in_dim, latent_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ModalityDecoder(nn.Module):
    """Decode latent features back to modality-specific feature space."""
    def __init__(self, latent_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class VectorQuantizer(nn.Module):
    """
    Optional vector quantization layer (codebook) for the latent space.
    Uses Gumbel-Softmax for differentiable discrete code selection.
    """
    def __init__(self, num_codes=8192, code_dim=256, temperature=1.0):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.temperature = temperature
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        """
        Args:
            z: [B, N, D] latent features

        Returns:
            z_q: [B, N, D] quantized features
            commit_loss: commitment loss for codebook training
        """
        B, N, D = z.shape
        # Compute distances to codebook entries
        flat_z = z.reshape(-1, D)  # [B*N, D]
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )  # [B*N, num_codes]

        # Soft assignment with Gumbel-Softmax
        soft_assign = F.gumbel_softmax(
            -distances, tau=self.temperature, hard=False)  # [B*N, num_codes]

        # Quantized features
        z_q = soft_assign @ self.codebook.weight  # [B*N, D]
        z_q = z_q.reshape(B, N, D)

        # Commitment loss
        commit_loss = F.mse_loss(z_q.detach(), z) + F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, commit_loss


class ModalityCompletion(nn.Module):
    """
    Masked Modality Completion module.

    For 2-modality case (RGB + IR):
    - When both available: compute latent alignment loss, pass original features
    - When one missing: reconstruct missing modality from the available one
    - Always compute reconstruction loss on available modalities

    Args:
        feature_dim: dimension of input features from backbone (e.g., 384 for ViT-S)
        latent_dim: dimension of the unified latent space
        num_modalities: number of modalities (default 2: RGB, IR)
        use_codebook: whether to use vector quantization
        codebook_size: number of codebook entries
    """
    def __init__(self, feature_dim=384, latent_dim=256, num_modalities=2,
                 use_codebook=False, codebook_size=8192):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        # Per-modality encoders and decoders
        self.encoders = nn.ModuleList([
            ModalityEncoder(feature_dim, latent_dim)
            for _ in range(num_modalities)
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(latent_dim, feature_dim)
            for _ in range(num_modalities)
        ])

        # Optional codebook
        self.use_codebook = use_codebook
        if use_codebook:
            self.codebooks = nn.ModuleList([
                VectorQuantizer(codebook_size, latent_dim)
                for _ in range(num_modalities)
            ])

    def forward(self, features, modality_mask):
        """
        Args:
            features: list of [B, N, d] tensors, one per modality
                features[0] = RGB branch features
                features[1] = IR branch features
            modality_mask: [B, num_modalities] binary mask
                1 = modality available, 0 = modality missing
                e.g., [1,1] = both available, [1,0] = RGB only, [0,1] = IR only

        Returns:
            completed_features: list of [B, N, d] tensors
                - original features for available modalities
                - reconstructed features for missing modalities
            losses: dict with 'latent_loss', 'rec_loss', 'commit_loss'
        """
        B = features[0].shape[0]
        device = features[0].device

        # Encode each modality to latent space
        latent_features = []
        commit_loss = torch.tensor(0.0, device=device)

        for i in range(self.num_modalities):
            z = self.encoders[i](features[i])  # [B, N, latent_dim]
            if self.use_codebook:
                z, c_loss = self.codebooks[i](z)
                commit_loss = commit_loss + c_loss
            latent_features.append(z)

        # --- Latent alignment loss ---
        # MSE between latent features of available modalities
        latent_loss = torch.tensor(0.0, device=device)
        n_latent_pairs = 0
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                # Only compute for samples where both modalities are available
                both_available = modality_mask[:, i] * modality_mask[:, j]  # [B]
                if both_available.sum() > 0:
                    mask = both_available.unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
                    diff = (latent_features[i] - latent_features[j]) * mask
                    latent_loss = latent_loss + diff.pow(2).sum() / (mask.sum() * latent_features[i].shape[1] * self.latent_dim)
                    n_latent_pairs += 1

        if n_latent_pairs > 0:
            latent_loss = latent_loss / n_latent_pairs

        # --- Reconstruct features for each modality ---
        completed_features = []
        rec_loss = torch.tensor(0.0, device=device)
        n_rec = 0

        for i in range(self.num_modalities):
            available_i = modality_mask[:, i]  # [B]

            # Compute aggregated latent from available modalities (excluding self)
            agg_latent = torch.zeros_like(latent_features[i])
            n_available = torch.zeros(B, 1, 1, device=device)
            for j in range(self.num_modalities):
                if j != i:
                    mask_j = modality_mask[:, j].unsqueeze(-1).unsqueeze(-1)
                    agg_latent = agg_latent + latent_features[j] * mask_j
                    n_available = n_available + mask_j

            # Avoid division by zero
            n_available = n_available.clamp(min=1)
            agg_latent = agg_latent / n_available

            # Decode reconstructed features
            recon_i = self.decoders[i](agg_latent)  # [B, N, d]

            # Compute reconstruction loss on available modalities
            if available_i.sum() > 0:
                mask_i = available_i.unsqueeze(-1).unsqueeze(-1)
                diff = (features[i] - recon_i) * mask_i
                rec_loss = rec_loss + diff.pow(2).sum() / (mask_i.sum() * features[i].shape[1] * self.feature_dim)
                n_rec += 1

            # Select features: original for available, reconstructed for missing
            mask_i = available_i.unsqueeze(-1).unsqueeze(-1).float()  # [B,1,1]
            completed = features[i] * mask_i + recon_i * (1 - mask_i)
            completed_features.append(completed)

        if n_rec > 0:
            rec_loss = rec_loss / n_rec

        losses = {
            'latent_loss': latent_loss,
            'rec_loss': rec_loss,
            'commit_loss': commit_loss if self.use_codebook else torch.tensor(0.0, device=device),
        }

        return completed_features, losses
