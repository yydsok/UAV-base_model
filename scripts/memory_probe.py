"""Quick memory probe: find max batch_size for vit_small/14 on single A800-80GB."""
import torch
import torch.cuda.amp as amp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.vision_transformer_rgbir import vit_small
from models.dino_head import DINOHead, MultiCropWrapper
from models.multi_granularity import MultiGranularityFeatures
from models.view_bridge import ViewDomainBridge

torch.backends.cudnn.benchmark = True


def _has_usable_cuda():
    try:
        return torch.cuda.is_available()
    except Exception as exc:
        print(f"CUDA check failed: {exc}")
        return False


if not _has_usable_cuda():
    print("CUDA is not available, skip memory probe.")
    sys.exit(0)

device = 'cuda'

# Build student (same as training)
backbone = vit_small(patch_size=14, in_chans=4, fusion='gated_cross_attn', drop_path_rate=0.1)
embed_dim = backbone.embed_dim  # 384
head = DINOHead(embed_dim, 65536, nlayers=3, hidden_dim=2048, bottleneck_dim=256)
student = MultiCropWrapper(backbone, head).to(device)

# Teacher (no grad)
backbone_t = vit_small(patch_size=14, in_chans=4, fusion='gated_cross_attn', drop_path_rate=0.0)
head_t = DINOHead(embed_dim, 65536, nlayers=3, hidden_dim=2048, bottleneck_dim=256)
teacher = MultiCropWrapper(backbone_t, head_t).to(device)
for p in teacher.parameters():
    p.requires_grad = False

# Auxiliary modules
mg = MultiGranularityFeatures(embed_dim, proj_dim=256, num_clusters=8).to(device)
vb = ViewDomainBridge(embed_dim, proj_dim=256, num_prototypes=64).to(device)

total_params = sum(p.numel() for p in student.parameters()) + \
               sum(p.numel() for p in mg.parameters()) + \
               sum(p.numel() for p in vb.parameters())
print(f"Student+aux params: {total_params/1e6:.1f}M")
print(f"Embed dim: {embed_dim}")

# Simulate training forward+backward at different batch sizes
n_global = 2
n_local = 8
n_crops = n_global + n_local

for bs in [256, 384, 512, 640, 768, 896, 1024]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(mg.parameters()) +
            list(vb.parameters()),
            lr=1e-4)
        scaler = amp.GradScaler()

        # Global crops: [B, 4, 224, 224] × 2
        # Local crops: [B, 4, 98, 98] × 8
        global_crops = [torch.randn(bs, 4, 224, 224, device=device) for _ in range(n_global)]
        local_crops = [torch.randn(bs, 4, 98, 98, device=device) for _ in range(n_local)]
        crops = global_crops + local_crops
        mask = torch.ones(bs, 2, device=device)

        with amp.autocast():
            teacher_out = teacher(crops[:2], modality_masks=mask)
            student_out, student_tokens = student(crops, return_backbone_feat=True, modality_masks=mask)

            # MG features
            B = bs
            global_tokens = student_tokens[:B*2]
            student_mg = mg(global_tokens)
            with torch.no_grad():
                t_tokens = []
                for gc in crops[:2]:
                    t_out = teacher.backbone(gc, return_all_tokens=True, modality_masks=mask)
                    t_tokens.append(t_out[:, 1:])
                teacher_mg_tokens = torch.cat(t_tokens, dim=0)
            teacher_mg = mg(teacher_mg_tokens)

            # Dummy loss
            loss = student_out.sum() * 0 + (student_mg['token'] - teacher_mg['token'].detach()).pow(2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  bs={bs:4d}  peak={peak_gb:.1f}GB  ({'OK' if peak_gb < 76 else 'TIGHT'})")

        del crops, global_crops, local_crops, student_out, student_tokens, loss
        del global_tokens, student_mg, teacher_mg, teacher_mg_tokens, t_tokens
        del optimizer, scaler
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"  bs={bs:4d}  OOM!")
            torch.cuda.empty_cache()
            break
        else:
            raise
