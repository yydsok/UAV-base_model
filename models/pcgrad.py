"""
PCGrad: Projected Conflicting Gradients — DINO-anchored variant.

For each auxiliary loss, compute its gradient independently.
If the auxiliary gradient conflicts with the DINO gradient (negative cosine
similarity), project it onto the normal plane of the DINO gradient before
accumulating. This ensures auxiliary tasks never degrade the primary DINO
self-distillation objective.

DDP-compatible: uses multiple .backward() calls so DDP hooks fire.
FP16-compatible: works with GradScaler.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
"""

import torch


def pcgrad_backward(dino_loss, aux_losses_dict, shared_params, fp16_scaler=None):
    """Project each auxiliary gradient against DINO gradient.

    Args:
        dino_loss: scalar tensor (DINO loss — anchor, never projected)
        aux_losses_dict: dict {name: (weight, loss_tensor)} for auxiliary losses
        shared_params: list of parameters to project gradients for
        fp16_scaler: optional GradScaler for mixed precision

    Returns:
        conflict_info: dict {name: cos_sim} for each auxiliary loss
    """
    # --- Step 1: Compute DINO gradient ---
    if fp16_scaler is not None:
        fp16_scaler.scale(dino_loss).backward(retain_graph=True)
    else:
        dino_loss.backward(retain_graph=True)

    # Save DINO gradient
    grad_dino = {}
    for p in shared_params:
        if p.grad is not None:
            grad_dino[p] = p.grad.clone()
        else:
            grad_dino[p] = torch.zeros_like(p.data)

    # Flatten DINO gradient for projection
    flat_dino = torch.cat([grad_dino[p].flatten() for p in shared_params])
    flat_dino_64 = flat_dino.double()
    norm_dino_sq = flat_dino_64.dot(flat_dino_64).clamp(min=1e-12)
    norm_dino = norm_dino_sq.sqrt()

    # --- Step 2: For each auxiliary loss, compute gradient and project ---
    conflict_info = {}
    accumulated_aux_grad = {p: torch.zeros_like(p.data) for p in shared_params}

    aux_items = list(aux_losses_dict.items())
    for idx, (name, (weight, loss)) in enumerate(aux_items):
        if not loss.requires_grad:
            conflict_info[name] = 0.0
            continue

        # Zero out grads before computing this aux gradient
        for p in shared_params:
            p.grad = None

        scaled_loss = weight * loss
        is_last = (idx == len(aux_items) - 1)
        if fp16_scaler is not None:
            fp16_scaler.scale(scaled_loss).backward(retain_graph=not is_last)
        else:
            scaled_loss.backward(retain_graph=not is_last)

        # Extract aux gradient
        grad_aux = {}
        for p in shared_params:
            if p.grad is not None:
                grad_aux[p] = p.grad.clone()
            else:
                grad_aux[p] = torch.zeros_like(p.data)

        # Compute cosine similarity with DINO gradient
        flat_aux = torch.cat([grad_aux[p].flatten() for p in shared_params])
        flat_aux_64 = flat_aux.double()
        if (not torch.isfinite(flat_aux_64).all()
                or not torch.isfinite(flat_dino_64).all()):
            # Drop only this auxiliary term for the current step; keep DINO anchor.
            conflict_info[name] = 0.0
            continue

        dot = flat_dino_64.dot(flat_aux_64)
        norm_aux = flat_aux_64.norm()

        if norm_aux < 1e-12:
            cos_sim = 0.0
        else:
            cos_sim_tensor = dot / (norm_dino * norm_aux)
            if torch.isfinite(cos_sim_tensor):
                cos_sim = cos_sim_tensor.item()
            else:
                conflict_info[name] = 0.0
                continue

        conflict_info[name] = cos_sim

        # If conflict: project auxiliary gradient onto DINO's normal plane
        if cos_sim < 0:
            # g_aux_proj = g_aux - (g_aux · g_dino / ||g_dino||^2) * g_dino
            proj_coeff = (dot / norm_dino_sq).to(flat_dino.dtype)
            for p in shared_params:
                grad_aux[p] = grad_aux[p] - proj_coeff * grad_dino[p]

        # Accumulate projected auxiliary gradient
        for p in shared_params:
            accumulated_aux_grad[p] = accumulated_aux_grad[p] + grad_aux[p]

    # --- Step 3: Set final gradient = DINO grad + accumulated projected aux grads ---
    for p in shared_params:
        p.grad = grad_dino[p] + accumulated_aux_grad[p]

    return conflict_info
