from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou


def _compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    changing_points = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[changing_points + 1] - recalls[changing_points]) * precisions[changing_points + 1])


def compute_detection_metrics(predictions, targets, num_classes):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_by_threshold = []
    ap50_by_class = {}

    for threshold in iou_thresholds:
        class_aps = []
        for class_id in range(1, num_classes + 1):
            gt_index = defaultdict(list)
            total_gt = 0
            for target in targets:
                image_id = int(target["image_id"].item())
                mask = target["labels"] == class_id
                gt_boxes = target["boxes"][mask]
                gt_index[image_id] = {
                    "boxes": gt_boxes,
                    "matched": torch.zeros((gt_boxes.shape[0],), dtype=torch.bool),
                }
                total_gt += gt_boxes.shape[0]

            class_preds = []
            for prediction in predictions:
                image_id = int(prediction["image_id"].item())
                mask = prediction["labels"] == class_id
                boxes = prediction["boxes"][mask]
                scores = prediction["scores"][mask]
                for score, box in zip(scores.tolist(), boxes):
                    class_preds.append((image_id, float(score), box.cpu()))
            class_preds.sort(key=lambda item: item[1], reverse=True)

            if total_gt == 0:
                continue
            if not class_preds:
                class_aps.append(0.0)
                if abs(threshold - 0.5) < 1e-6:
                    ap50_by_class[class_id] = 0.0
                continue

            true_positives = np.zeros((len(class_preds),), dtype=np.float32)
            false_positives = np.zeros((len(class_preds),), dtype=np.float32)

            for pred_idx, (image_id, _, pred_box) in enumerate(class_preds):
                gt_entry = gt_index.get(image_id)
                if gt_entry is None or gt_entry["boxes"].numel() == 0:
                    false_positives[pred_idx] = 1.0
                    continue

                ious = box_iou(pred_box.unsqueeze(0), gt_entry["boxes"]).squeeze(0)
                best_iou, best_gt_idx = torch.max(ious, dim=0)
                if best_iou.item() >= threshold and not gt_entry["matched"][best_gt_idx]:
                    true_positives[pred_idx] = 1.0
                    gt_entry["matched"][best_gt_idx] = True
                else:
                    false_positives[pred_idx] = 1.0

            tp_cum = np.cumsum(true_positives)
            fp_cum = np.cumsum(false_positives)
            recalls = tp_cum / max(total_gt, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap = _compute_ap(recalls, precisions)
            class_aps.append(ap)
            if abs(threshold - 0.5) < 1e-6:
                ap50_by_class[class_id] = ap

        ap_by_threshold.append(np.mean(class_aps) if class_aps else 0.0)

    map50 = ap_by_threshold[0] if ap_by_threshold else 0.0
    return {
        "mAP50": float(map50 * 100.0),
        "mAP": float(np.mean(ap_by_threshold) * 100.0 if ap_by_threshold else 0.0),
        "AP50_per_class": {int(k): float(v * 100.0) for k, v in ap50_by_class.items()},
    }


def compute_change_detection_metrics(predictions, targets):
    tp = fp = tn = fn = 0
    for pred, tgt in zip(predictions, targets):
        pred_flat = pred.reshape(-1).astype(np.int64)
        tgt_flat = tgt.reshape(-1).astype(np.int64)
        valid = tgt_flat >= 0
        pred_flat = pred_flat[valid]
        tgt_flat = tgt_flat[valid]

        tp += int(((pred_flat == 1) & (tgt_flat == 1)).sum())
        fp += int(((pred_flat == 1) & (tgt_flat == 0)).sum())
        tn += int(((pred_flat == 0) & (tgt_flat == 0)).sum())
        fn += int(((pred_flat == 0) & (tgt_flat == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    oa = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "F1": float(f1 * 100.0),
        "IoU": float(iou * 100.0),
        "Precision": float(precision * 100.0),
        "Recall": float(recall * 100.0),
        "OA": float(oa * 100.0),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


class SegmentationMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target):
        pred = prediction.reshape(-1)
        tgt = target.reshape(-1)
        valid = tgt != self.ignore_index
        pred = pred[valid]
        tgt = tgt[valid]
        bincount = np.bincount(
            self.num_classes * tgt.astype(np.int64) + pred.astype(np.int64),
            minlength=self.num_classes ** 2,
        )
        self.confusion += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self):
        diag = np.diag(self.confusion)
        gt_sum = self.confusion.sum(axis=1)
        pred_sum = self.confusion.sum(axis=0)
        union = gt_sum + pred_sum - diag
        iou = diag / np.maximum(union, 1)
        acc = diag.sum() / np.maximum(self.confusion.sum(), 1)
        return {
            "mIoU": float(np.nanmean(iou) * 100.0),
            "pixel_acc": float(acc * 100.0),
            "IoU_per_class": (iou * 100.0).tolist(),
        }


class IouTracker:
    def __init__(self, iou_threshold=0.3, max_age=20, min_hits=2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        for track in self.tracks:
            track["age"] += 1

        if self.tracks and detections:
            cost = np.ones((len(self.tracks), len(detections)), dtype=np.float32)
            for track_idx, track in enumerate(self.tracks):
                track_box = torch.tensor(track["box"], dtype=torch.float32).unsqueeze(0)
                for det_idx, det in enumerate(detections):
                    if det["label"] != track["label"]:
                        continue
                    det_box = torch.tensor(det["box"], dtype=torch.float32).unsqueeze(0)
                    iou = box_iou(track_box, det_box)[0, 0].item()
                    cost[track_idx, det_idx] = 1.0 - iou

            track_indices, det_indices = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_dets = set()
            for track_idx, det_idx in zip(track_indices, det_indices):
                iou = 1.0 - cost[track_idx, det_idx]
                if iou < self.iou_threshold:
                    continue
                self.tracks[track_idx]["box"] = detections[det_idx]["box"]
                self.tracks[track_idx]["score"] = detections[det_idx]["score"]
                self.tracks[track_idx]["label"] = detections[det_idx]["label"]
                self.tracks[track_idx]["age"] = 0
                self.tracks[track_idx]["hits"] += 1
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        else:
            matched_dets = set()

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            self.tracks.append(
                {
                    "track_id": self.next_id,
                    "box": det["box"],
                    "label": det["label"],
                    "score": det["score"],
                    "age": 0,
                    "hits": 1,
                }
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track["age"] <= self.max_age]
        outputs = []
        for track in self.tracks:
            if track["hits"] >= self.min_hits or track["age"] == 0:
                outputs.append(track.copy())
        return outputs


def _match_tracks_to_detections(tracks, detections, iou_threshold):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.ones((len(tracks), len(detections)), dtype=np.float32)
    for track_idx, track in enumerate(tracks):
        track_box = torch.tensor(track["box"], dtype=torch.float32).unsqueeze(0)
        for det_idx, det in enumerate(detections):
            if det["label"] != track["label"]:
                continue
            det_box = torch.tensor(det["box"], dtype=torch.float32).unsqueeze(0)
            iou = box_iou(track_box, det_box)[0, 0].item()
            cost[track_idx, det_idx] = 1.0 - iou

    track_indices, det_indices = linear_sum_assignment(cost)
    matches = []
    matched_tracks = set()
    matched_dets = set()
    for track_idx, det_idx in zip(track_indices, det_indices):
        iou = 1.0 - cost[track_idx, det_idx]
        if iou < iou_threshold:
            continue
        matches.append((track_idx, det_idx))
        matched_tracks.add(track_idx)
        matched_dets.add(det_idx)
    unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
    unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class ByteTrackStyleTracker:
    """A lightweight ByteTrack-style tracker with two-stage IoU association."""

    def __init__(self, iou_threshold=0.3, high_threshold=0.5, low_threshold=0.1, max_age=20, min_hits=2):
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = []

    def _update_track(self, track, det):
        track["box"] = det["box"]
        track["score"] = det["score"]
        track["label"] = det["label"]
        track["age"] = 0
        track["hits"] += 1

    def _new_track(self, det):
        self.tracks.append(
            {
                "track_id": self.next_id,
                "box": det["box"],
                "label": det["label"],
                "score": det["score"],
                "age": 0,
                "hits": 1,
            }
        )
        self.next_id += 1

    def update(self, detections):
        for track in self.tracks:
            track["age"] += 1

        high_dets = [det for det in detections if det["score"] >= self.high_threshold]
        low_dets = [det for det in detections if self.low_threshold <= det["score"] < self.high_threshold]

        matches, unmatched_tracks, unmatched_high = _match_tracks_to_detections(
            self.tracks, high_dets, self.iou_threshold
        )
        for track_idx, det_idx in matches:
            self._update_track(self.tracks[track_idx], high_dets[det_idx])

        remaining_tracks = [self.tracks[idx] for idx in unmatched_tracks]
        low_matches, _still_unmatched_tracks, _unmatched_low = _match_tracks_to_detections(
            remaining_tracks, low_dets, self.iou_threshold
        )
        for rel_track_idx, det_idx in low_matches:
            track_idx = unmatched_tracks[rel_track_idx]
            self._update_track(self.tracks[track_idx], low_dets[det_idx])

        for det_idx in unmatched_high:
            self._new_track(high_dets[det_idx])

        self.tracks = [track for track in self.tracks if track["age"] <= self.max_age]
        outputs = []
        for track in self.tracks:
            if track["hits"] >= self.min_hits or track["age"] == 0:
                outputs.append(track.copy())
        return outputs


def evaluate_mot_sequence(
    sequence,
    detector_fn,
    score_threshold=0.3,
    iou_threshold=0.5,
    tracker_type="iou",
    tracker_kwargs=None,
    external_tracker_fn=None,
):
    tracker_kwargs = tracker_kwargs or {}
    tracker = None
    if external_tracker_fn is None:
        if tracker_type == "iou":
            tracker = IouTracker(
                iou_threshold=tracker_kwargs.get("iou_threshold", 0.3),
                max_age=tracker_kwargs.get("max_age", 20),
                min_hits=tracker_kwargs.get("min_hits", 2),
            )
        elif tracker_type == "bytetrack":
            tracker = ByteTrackStyleTracker(
                iou_threshold=tracker_kwargs.get("iou_threshold", 0.3),
                high_threshold=tracker_kwargs.get("high_threshold", 0.5),
                low_threshold=tracker_kwargs.get("low_threshold", 0.1),
                max_age=tracker_kwargs.get("max_age", 20),
                min_hits=tracker_kwargs.get("min_hits", 2),
            )
        else:
            raise ValueError(f"Unsupported tracker type '{tracker_type}'.")

    total_gt = 0
    total_pred = 0
    false_negatives = 0
    false_positives = 0
    id_switches = 0
    overlap_counts = defaultdict(int)
    last_match = {}

    for frame in sequence["frames"]:
        detections = detector_fn(frame)
        detections = [det for det in detections if det["score"] >= score_threshold]
        if external_tracker_fn is not None:
            tracks = external_tracker_fn(frame, detections)
        else:
            tracks = tracker.update(detections)

        gt_ann = frame["annotations"]
        total_gt += len(gt_ann)
        total_pred += len(tracks)

        if gt_ann and tracks:
            gt_boxes = torch.tensor([item["box"] for item in gt_ann], dtype=torch.float32)
            track_boxes = torch.tensor([item["box"] for item in tracks], dtype=torch.float32)
            ious = box_iou(gt_boxes, track_boxes).numpy()
            cost = 1.0 - ious
            gt_idx, track_idx = linear_sum_assignment(cost)

            matched_gt = set()
            matched_tracks = set()
            for g_idx, t_idx in zip(gt_idx, track_idx):
                if ious[g_idx, t_idx] < iou_threshold:
                    continue
                gt_item = gt_ann[g_idx]
                track_item = tracks[t_idx]
                matched_gt.add(g_idx)
                matched_tracks.add(t_idx)
                overlap_counts[(gt_item["track_id"], track_item["track_id"])] += 1
                prev_track = last_match.get(gt_item["track_id"])
                if prev_track is not None and prev_track != track_item["track_id"]:
                    id_switches += 1
                last_match[gt_item["track_id"]] = track_item["track_id"]

            false_negatives += len(gt_ann) - len(matched_gt)
            false_positives += len(tracks) - len(matched_tracks)
        else:
            false_negatives += len(gt_ann)
            false_positives += len(tracks)

    gt_ids = sorted({item[0] for item in overlap_counts.keys()})
    pred_ids = sorted({item[1] for item in overlap_counts.keys()})
    if gt_ids and pred_ids:
        matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        gt_lookup = {track_id: idx for idx, track_id in enumerate(gt_ids)}
        pred_lookup = {track_id: idx for idx, track_id in enumerate(pred_ids)}
        for (gt_id, pred_id), value in overlap_counts.items():
            matrix[gt_lookup[gt_id], pred_lookup[pred_id]] = value
        assignment = matrix.max() - matrix
        gt_idx, pred_idx = linear_sum_assignment(assignment)
        idtp = sum(matrix[g, p] for g, p in zip(gt_idx, pred_idx))
    else:
        idtp = 0

    idfp = total_pred - idtp
    idfn = total_gt - idtp
    mota = 100.0 * (1.0 - (false_negatives + false_positives + id_switches) / max(total_gt, 1))
    idf1 = 100.0 * (2 * idtp) / max(2 * idtp + idfp + idfn, 1)
    return {
        "MOTA": float(mota),
        "IDF1": float(idf1),
        "FP": int(false_positives),
        "FN": int(false_negatives),
        "IDSW": int(id_switches),
        "GT": int(total_gt),
        "IDTP": int(idtp),
        "IDFP": int(idfp),
        "IDFN": int(idfn),
    }
