#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.datasets import _merge_rgb_ir, build_tracking_dataset
from downstream.metrics import evaluate_mot_sequence
from downstream.models import build_detection_model
from downstream.openmmlab import MMByteTrackAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOT using a DINO-MM detector")
    parser.add_argument("--dataset", default="m3ot", choices=["m3ot", "vt_tiny_mot", "visdrone_mot"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--detector_checkpoint", default=None,
                        help="checkpoint_best.pth produced by train_detection.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tracker", default="bytetrack", choices=["iou", "bytetrack", "mm_bytetrack"])
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--track_iou_thresh", type=float, default=0.3)
    parser.add_argument("--track_high_thresh", type=float, default=0.5)
    parser.add_argument("--track_low_thresh", type=float, default=0.1)
    parser.add_argument("--max_age", type=int, default=20)
    parser.add_argument("--min_hits", type=int, default=2)
    parser.add_argument("--use_gt_detections", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    if not args.use_gt_detections and not args.detector_checkpoint:
        parser.error("--detector_checkpoint is required unless --use_gt_detections is set.")
    return args


def resolve_tracking_split(dataset_name):
    if dataset_name == "vt_tiny_mot":
        return "test"
    return "val"


def load_detector(task_checkpoint_path, device):
    state = torch.load(task_checkpoint_path, map_location="cpu", weights_only=False)
    config = state["task_config"]
    model, _ = build_detection_model(
        config["checkpoint"],
        num_classes=config["num_classes"],
        checkpoint_key=config.get("checkpoint_key", "teacher"),
        device=device,
        init_mode=config.get("init_mode", "pretrained"),
        arch=config.get("arch"),
        patch_size=config.get("patch_size"),
        in_chans=config.get("in_chans"),
        fusion=config.get("fusion"),
        trainable_blocks=config.get("trainable_blocks", 4),
        feature_dim=config.get("feature_dim", 256),
        min_size=config.get("min_size", 640),
        max_size=config.get("max_size", 1333),
        score_thresh=config.get("score_thresh", 0.05),
        framework=config.get("framework", "fasterrcnn"),
    )
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    return model, config


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tracking_dataset = build_tracking_dataset(args.dataset, args.data_root, split=resolve_tracking_split(args.dataset))
    detector = None
    detector_config = {}
    if not args.use_gt_detections:
        detector, detector_config = load_detector(args.detector_checkpoint, device)

    all_results = {}
    total_fp = total_fn = total_idsw = total_gt = 0
    total_idtp = total_idfp = total_idfn = 0

    for sequence in tracking_dataset.sequences:
        if args.max_frames is not None:
            sequence = {
                **sequence,
                "frames": sequence["frames"][:args.max_frames],
            }

        if args.use_gt_detections:
            def detector_fn(frame):
                return [
                    {
                        "box": ann["box"],
                        "score": 1.0,
                        "label": ann["label"],
                    }
                    for ann in frame["annotations"]
                ]
        else:
            modality = detector_config.get("modality", "both")

            @torch.no_grad()
            def detector_fn(frame):
                image = _merge_rgb_ir(frame["rgb_path"], frame["ir_path"], modality=modality).to(device)
                output = detector([image])[0]
                detections = []
                for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                    detections.append(
                        {
                            "box": box.cpu().tolist(),
                            "score": float(score.item()),
                            "label": int(label.item()),
                        }
                    )
                return detections

        external_tracker_fn = None
        if args.tracker == "mm_bytetrack":
            mm_tracker = MMByteTrackAdapter(
                modality=detector_config.get("modality", "both"),
                high_threshold=args.track_high_thresh,
                low_threshold=args.track_low_thresh,
                match_iou_threshold=args.track_iou_thresh,
                init_track_threshold=args.track_high_thresh,
                tentative_frames=max(1, args.min_hits),
                device=device,
            )
            external_tracker_fn = mm_tracker.update_frame

        metrics = evaluate_mot_sequence(
            sequence,
            detector_fn=detector_fn,
            score_threshold=args.score_thresh,
            iou_threshold=args.iou_thresh,
            tracker_type="bytetrack" if args.tracker == "mm_bytetrack" else args.tracker,
            tracker_kwargs={
                "iou_threshold": args.track_iou_thresh,
                "high_threshold": args.track_high_thresh,
                "low_threshold": args.track_low_thresh,
                "max_age": args.max_age,
                "min_hits": args.min_hits,
            },
            external_tracker_fn=external_tracker_fn,
        )
        all_results[sequence["name"]] = metrics
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]
        total_idsw += metrics["IDSW"]
        total_gt += metrics["GT"]
        total_idtp += metrics["IDTP"]
        total_idfp += metrics["IDFP"]
        total_idfn += metrics["IDFN"]
        print(f"{sequence['name']}: MOTA={metrics['MOTA']:.2f} IDF1={metrics['IDF1']:.2f}")

    summary = {
        "MOTA": 100.0 * (1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)),
        "IDF1": 100.0 * (2 * total_idtp) / max(2 * total_idtp + total_idfp + total_idfn, 1),
        "FP": total_fp,
        "FN": total_fn,
        "IDSW": total_idsw,
        "GT": total_gt,
        "IDTP": total_idtp,
        "IDFP": total_idfp,
        "IDFN": total_idfn,
        "per_sequence": all_results,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
