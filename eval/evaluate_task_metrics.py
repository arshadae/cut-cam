#!/usr/bin/env python3
import argparse, json, os, sys, time, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

# -------------------------
# Utilities
# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder: str):
    return [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXTS]

def pil_loader(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB")

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:120]

# -------------------------
# Model loader (user-provided)
# -------------------------
def load_user_model(model_py: str, device: str):
    """
    model_py: path to a python file that defines:
        def load_model(device:str):
            # return (torch.nn.Module in eval mode, transform_fn(PIL)->tensor)
            return model, transform
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    model, transform = mod.load_model(device)
    model.to(device).eval()
    return model, transform

# -------------------------
# Classification metrics
# -------------------------
def evaluate_classification(img_dir: str, labels_csv: str, model, transform, device: str,
                            batch_size: int = 32, out_json: Optional[Path] = None):
    """
    labels_csv: CSV with header: filename,label
    """
    import csv
    # Load labels
    y_true, paths = [], []
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(img_dir) / row["filename"]
            if p.exists():
                paths.append(p)
                y_true.append(int(row["label"]))
    if not paths:
        raise RuntimeError("No labeled images found matching CSV and img_dir.")
    y_true = np.array(y_true)

    # Inference
    preds = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Classification inference"):
        imgs = [pil_loader(p) for p in paths[i:i+batch_size]]
        x = torch.stack([transform(im) for im in imgs]).to(device)
        with torch.no_grad():
            logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
    y_pred = np.concatenate(preds, axis=0)

    # Metrics
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    acc = (y_true == y_pred).mean().item()
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        precision.append(p); recall.append(r); f1.append(f)

    result = {
        "task": "classification",
        "accuracy": float(acc),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "confusion_matrix": conf.tolist(),
        "num_samples": len(y_true),
        "num_classes": num_classes
    }
    if out_json:
        out_json.write_text(json.dumps(result, indent=2))
    return result

# -------------------------
# Segmentation metrics (semantic)
# -------------------------
def fast_hist(pred, gt, num_classes, ignore_index=None):
    mask = (gt >= 0) & (gt < num_classes)
    if ignore_index is not None:
        mask &= (gt != ignore_index)
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def evaluate_segmentation(img_dir: str, mask_dir: str, model, transform, device: str,
                          num_classes: int, ignore_index: Optional[int],
                          batch_size: int = 4, out_json: Optional[Path] = None,
                          mask_suffix: str = ".png"):
    """
    mask_dir: folder with per-pixel integer masks named like image stem + mask_suffix
    """
    img_paths = list_images(img_dir)
    if not img_paths:
        raise RuntimeError("No images for segmentation.")
    hist_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_pix, correct_pix = 0, 0

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Segmentation inference"):
        batch_paths = img_paths[i:i+batch_size]
        imgs = [pil_loader(p) for p in batch_paths]
        x = torch.stack([transform(im) for im in imgs]).to(device)
        with torch.no_grad():
            logits = model(x)  # (B, C, H, W)
        pred = logits.argmax(1).cpu().numpy()

        for b, p in enumerate(batch_paths):
            mask_path = Path(mask_dir) / (p.stem + mask_suffix)
            if not mask_path.exists():
                continue
            gt = np.array(Image.open(mask_path), dtype=np.int64)
            ph, pw = pred[b].shape
            if gt.shape != (ph, pw):
                gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((pw, ph), resample=Image.NEAREST), dtype=np.int64)
            hist_total += fast_hist(pred[b], gt, num_classes, ignore_index)
            if ignore_index is not None:
                m = (gt != ignore_index)
                correct_pix += np.sum((pred[b] == gt) & m)
                total_pix += int(m.sum())
            else:
                correct_pix += int((pred[b] == gt).sum())
                total_pix += gt.size

    iu = np.diag(hist_total) / (hist_total.sum(1) + hist_total.sum(0) - np.diag(hist_total) + 1e-12)
    miou = float(np.nanmean(iu))
    pix_acc = float(correct_pix / (total_pix + 1e-12))

    result = {
        "task": "segmentation",
        "mean_IoU": miou,
        "per_class_IoU": iu.tolist(),
        "pixel_accuracy": pix_acc,
        "num_images": len(img_paths),
        "num_classes": num_classes,
        "ignore_index": ignore_index
    }
    if out_json:
        out_json.write_text(json.dumps(result, indent=2))
    return result

# -------------------------
# Detection metrics (COCO mAP)
# -------------------------
def evaluate_detection_coco(img_dir: str, coco_gt_json: str, model, transform, device: str,
                            score_thresh: float = 0.0, batch_size: int = 1,
                            out_json: Optional[Path] = None):
    """
    Assumes the model returns torchvision-style detections:
      list[dict(boxes: Nx4 xyxy, labels: N, scores: N)]
    Ground truth must be COCO JSON.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(coco_gt_json)
    img_ids = coco_gt.getImgIds()
    id_to_file = {i["id"]: i["file_name"] for i in coco_gt.loadImgs(img_ids)}
    # filter to images that exist in img_dir
    valid_ids = [i for i in img_ids if (Path(img_dir) / id_to_file[i]).exists()]

    detections = []
    for idx in tqdm(range(0, len(valid_ids), batch_size), desc="Detection inference"):
        bid = valid_ids[idx:idx+batch_size]
        ims = [pil_loader(Path(img_dir) / id_to_file[i]) for i in bid]
        x = torch.stack([transform(im) for im in ims]).to(device)
        with torch.no_grad():
            out = model(x)
        # ensure list
        if isinstance(out, dict):
            out = [out]
        for i, oid in enumerate(bid):
            det = out[i]
            boxes = det["boxes"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            labels = det["labels"].detach().cpu().numpy()
            # xyxy -> xywh
            xywh = boxes.copy()
            xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
            xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
            for j in range(xywh.shape[0]):
                if scores[j] < score_thresh: 
                    continue
                detections.append({
                    "image_id": int(oid),
                    "category_id": int(labels[j]),
                    "bbox": [float(xywh[j,0]), float(xywh[j,1]), float(xywh[j,2]), float(xywh[j,3])],
                    "score": float(scores[j])
                })

    # Evaluate
    if len(detections) == 0:
        raise RuntimeError("No detections produced; check model outputs and score_thresh.")
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = coco_eval.stats  # 12-element vector
    result = {
        "task": "detection",
        "COCO_metrics": {
            "AP@[0.50:0.95]": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
            "AR_1": float(stats[6]),
            "AR_10": float(stats[7]),
            "AR_100": float(stats[8]),
            "AR_small": float(stats[9]),
            "AR_medium": float(stats[10]),
            "AR_large": float(stats[11]),
        },
        "num_images": len(valid_ids),
        "score_thresh": score_thresh
    }
    if out_json:
        out_json.write_text(json.dumps(result, indent=2))
    return result

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Task-level evaluation")
    ap.add_argument("--task", required=True, choices=["classification", "segmentation", "detection"])
    ap.add_argument("--model_py", required=True, help="Path to model_def.py with load_model(device)->(model, transform)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--eval_name", type=str, default=None)

    # shared
    ap.add_argument("--img_dir", type=str, required=True)

    # classification
    ap.add_argument("--labels_csv", type=str, help="CSV (filename,label) for classification")

    # segmentation
    ap.add_argument("--mask_dir", type=str, help="Folder with integer masks for segmentation")
    ap.add_argument("--num_classes", type=int, help="Number of classes for segmentation")
    ap.add_argument("--ignore_index", type=int, default=None)
    ap.add_argument("--mask_suffix", type=str, default=".png")

    # detection
    ap.add_argument("--coco_gt", type=str, help="COCO ground truth JSON for detection")
    ap.add_argument("--score_thresh", type=float, default=0.0)

    args = ap.parse_args()

    out_json = Path(args.out_json) if args.out_json else None
    if out_json and args.eval_name:
        out_json = out_json.with_name(out_json.stem + f"_{slug(args.eval_name)}" + out_json.suffix)

    model, transform = load_user_model(args.model_py, args.device)

    if args.task == "classification":
        if not args.labels_csv:
            raise ValueError("--labels_csv required for classification")
        res = evaluate_classification(args.img_dir, args.labels_csv, model, transform, args.device,
                                      batch_size=args.batch_size, out_json=out_json)

    elif args.task == "segmentation":
        if not args.mask_dir or args.num_classes is None:
            raise ValueError("--mask_dir and --num_classes required for segmentation")
        res = evaluate_segmentation(args.img_dir, args.mask_dir, model, transform, args.device,
                                    num_classes=args.num_classes, ignore_index=args.ignore_index,
                                    batch_size=args.batch_size, out_json=out_json, mask_suffix=args.mask_suffix)

    else:  # detection
        if not args.coco_gt:
            raise ValueError("--coco_gt required for detection")
        res = evaluate_detection_coco(args.img_dir, args.coco_gt, model, transform, args.device,
                                      score_thresh=args.score_thresh, batch_size=1, out_json=out_json)

    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

# Example usage
# Classification:
# python evaluate_task_metrics.py \
#   --task classification \
#   --model_py model_def.py \
#   --img_dir /path/to/images \
#   --labels_csv /path/to/labels.csv \
#   --out_json ./cls_metrics.json \
#   --eval_name run1

# Segmentation:
# python evaluate_task_metrics.py \
#   --task segmentation \
#   --model_py model_def.py \
#   --img_dir /path/to/images \
#   --mask_dir /path/to/masks \
#   --num_classes 7 \
#   --ignore_index 255 \
#   --out_json ./seg_metrics.json \
#   --eval_name run1

# Detection (COCO):
# python evaluate_task_metrics.py \
#   --task detection \
#   --model_py model_def.py \
#   --img_dir /path/to/images \
#   --coco_gt /path/to/instances_val.json \
#   --out_json ./det_metrics.json \
#   --eval_name run1

# Requirements:
# pycocotools>=2.0.7   # detection only
