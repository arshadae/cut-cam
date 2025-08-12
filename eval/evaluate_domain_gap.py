#!/usr/bin/env python3
import argparse, os, json, math, random, warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from packaging import version
import sklearn

import re
def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:120]

from typing import Optional
import csv, re


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import lpips  # type: ignore
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

from scipy import linalg

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Utilities
# ---------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder, max_images=None):
    paths = [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if max_images and len(paths) > max_images:
        paths = paths[:max_images]
    return paths

def pil_loader(path):
    with Image.open(path) as im:
        return im.convert("RGB")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ---------------------------
# Feature extractors
# ---------------------------
class InceptionFeat(nn.Module):
    """InceptionV3 pool3/logits (2048-D) features for FID/KID/PAD/MMD/etc., version-robust."""
    def __init__(self):
        super().__init__()
        # Load weights + required transform in a version-safe way
        try:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            net = models.inception_v3(weights=weights, aux_logits=True)
            tf = weights.transforms()   # <-- use the canonical preprocessing
        except Exception:
            # Fallback for older/newer APIs
            net = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)  # or pretrained=True in very old
            tf = transforms.Compose([
                transforms.Resize(342),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # Expose 2048-D features
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        self.backbone = net
        self.tf = tf

    @torch.no_grad()
    def forward(self, imgs):  # imgs: list of PIL.Image
        x = torch.stack([self.tf(im) for im in imgs], dim=0).to(self.device)
        out = self.backbone(x)
        # Normalize output across torchvision versions
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return out  # N x 2048

    @property
    def device(self):
        return next(self.backbone.parameters()).device


def load_in_batches(paths, model, batch_size=32):
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="Featurizing", leave=False):
            batch = [pil_loader(p) for p in paths[i:i+batch_size]]
            f = model(batch)
            feats.append(f.cpu().numpy())
    if len(feats) == 0:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(feats, axis=0)

# Task Model Loader
def _load_user_model(model_py: str, device: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_task_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    model, transform = mod.load_model(device)
    model.to(device).eval()
    return model, transform

# Classification Metrics 
def _eval_classification(img_dir: str, labels_csv: str, model, transform, device: str,
                         batch_size: int = 32):
    paths, y_true = [], []
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(img_dir) != Path(img_dir) and None or None  # no-op to keep lints happy
            p = Path(img_dir) / row["filename"]
            if p.exists():
                paths.append(p)
                y_true.append(int(row["label"]))
    if not paths:
        raise RuntimeError("No labeled images matched filenames in labels_csv.")
    y_true = np.array(y_true)

    preds = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Cls inference", leave=False):
        imgs = [pil_loader(p) for p in paths[i:i+batch_size]]
        x = torch.stack([transform(im) for im in imgs]).to(device)
        with torch.no_grad():
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds, 0)

    C = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    acc = float((y_true == y_pred).mean())
    precision, recall, f1 = [], [], []
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        precision.append(float(p)); recall.append(float(r)); f1.append(float(f))

    return {
        "task": "classification",
        "accuracy": acc,
        "per_class": {"precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix": conf.tolist(),
        "num_samples": int(len(y_true)),
        "num_classes": int(C),
    }

# Segmentation Metrics
def _fast_hist(pred, gt, C, ignore_index=None):
    mask = (gt >= 0) & (gt < C)
    if ignore_index is not None:
        mask &= (gt != ignore_index)
    return np.bincount(C*gt[mask].astype(int)+pred[mask].astype(int), minlength=C*C).reshape(C, C)

def _eval_segmentation(img_dir: str, mask_dir: str, model, transform, device: str,
                       num_classes: int, ignore_index: Optional[int], mask_suffix=".png",
                       batch_size: int = 4):
    img_paths = [p for p in sorted(Path(img_dir).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not img_paths:
        raise RuntimeError("No images for segmentation.")
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_pix = correct_pix = 0

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Seg inference", leave=False):
        batch = img_paths[i:i+batch_size]
        ims = [pil_loader(p) for p in batch]
        x = torch.stack([transform(im) for im in ims]).to(device)
        with torch.no_grad():
            logits = model(x)  # B,C,H,W
        pred = logits.argmax(1).cpu().numpy()
        for b, p in enumerate(batch):
            gt_path = Path(mask_dir) / (p.stem + mask_suffix)
            if not gt_path.exists():
                continue
            gt = np.array(Image.open(gt_path), dtype=np.int64)
            ph, pw = pred[b].shape
            if gt.shape != (ph, pw):
                gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((pw, ph), resample=Image.NEAREST), dtype=np.int64)
            hist += _fast_hist(pred[b], gt, num_classes, ignore_index)
            if ignore_index is not None:
                m = (gt != ignore_index)
                correct_pix += int(((pred[b] == gt) & m).sum())
                total_pix += int(m.sum())
            else:
                correct_pix += int((pred[b] == gt).sum())
                total_pix += gt.size

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)
    miou = float(np.nanmean(iu))
    pix_acc = float(correct_pix / (total_pix + 1e-12))
    return {
        "task": "segmentation",
        "mean_IoU": miou,
        "per_class_IoU": iu.tolist(),
        "pixel_accuracy": pix_acc,
        "num_images": int(len(img_paths)),
        "num_classes": int(num_classes),
        "ignore_index": (None if ignore_index is None else int(ignore_index)),
    }


# Detection Metrics
def _eval_detection_coco(img_dir: str, coco_gt_json: str, model, transform, device: str,
                         score_thresh: float = 0.0, batch_size: int = 1):
    # ---- NumPy compatibility for older pycocotools ----
    import numpy as _np
    if not hasattr(_np, "float"):
        _np.float = float
    if not hasattr(_np, "int"):
        _np.int = int
    if not hasattr(_np, "bool"):
        _np.bool = bool
    if not hasattr(_np, "object"):
        _np.object = object
    if not hasattr(_np, "long"):
        _np.long = int
    # ---------------------------------------------------
    
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(coco_gt_json)
    img_ids = coco_gt.getImgIds()
    id2file = {i["id"]: i["file_name"] for i in coco_gt.loadImgs(img_ids)}
    valid_ids = [i for i in img_ids if (Path(img_dir) / id2file[i]).exists()]

    dets = []
    for idx in tqdm(range(0, len(valid_ids), batch_size), desc="Det inference", leave=False):
        bid = valid_ids[idx:idx+batch_size]
        ims = [pil_loader(Path(img_dir) / id2file[i]) for i in bid]
        x = torch.stack([transform(im) for im in ims]).to(device)
        with torch.no_grad():
            out = model(x)
        if isinstance(out, dict):
            out = [out]
        for i, oid in enumerate(bid):
            boxes = out[i]["boxes"].detach().cpu().numpy()
            scores = out[i]["scores"].detach().cpu().numpy()
            labels = out[i]["labels"].detach().cpu().numpy()
            xywh = boxes.copy()
            xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
            xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
            for j in range(xywh.shape[0]):
                if scores[j] < score_thresh:
                    continue
                dets.append({
                    "image_id": int(oid),
                    "category_id": int(labels[j]),
                    "bbox": [float(xywh[j,0]), float(xywh[j,1]), float(xywh[j,2]), float(xywh[j,3])],
                    "score": float(scores[j])
                })
    if not dets:
        raise RuntimeError("No detections produced; check model and --score_thresh.")
    coco_dt = coco_gt.loadRes(dets)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return {
        "task": "detection",
        "COCO_metrics": {
            "AP@[0.50:0.95]": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
            "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
            "AR_1": float(s[6]), "AR_10": float(s[7]), "AR_100": float(s[8]),
            "AR_small": float(s[9]), "AR_medium": float(s[10]), "AR_large": float(s[11]),
        },
        "num_images": int(len(valid_ids)),
        "score_thresh": float(score_thresh),
    }

    

# ---------------------------
# Metrics
# ---------------------------
def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((sigma1 + eps*np.eye(sigma1.shape[0])) @ (sigma2 + eps*np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return float(fid)

def compute_kid(feats1, feats2, n_subsets=100, subset_size=1000):
    """KID: unbiased MMD^2 with polynomial kernel k(x,y)=(x.y/ d + 1)^3"""
    n1, n2 = feats1.shape[0], feats2.shape[0]
    d = feats1.shape[1]
    if n1 < subset_size or n2 < subset_size:
        subset_size = min(n1, n2)
    kids = []
    for _ in range(n_subsets):
        x = feats1[np.random.choice(n1, subset_size, replace=False)]
        y = feats2[np.random.choice(n2, subset_size, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        m = subset_size
        kid = (kxx.sum()/(m*(m-1)) + kyy.sum()/(m*(m-1)) - 2*kxy.mean())
        kids.append(kid)
    return float(np.mean(kids)), float(np.std(kids))

def compute_color_hist_distance(paths_a, paths_b, bins=64):
    """Chi-square distance between mean RGB histograms per domain (normalized)."""
    def mean_hist(paths):
        hsum = np.zeros((3, bins), dtype=np.float64)
        for p in tqdm(paths, desc="Hist", leave=False):
            arr = np.asarray(pil_loader(p).resize((256,256)), dtype=np.uint8)
            for c in range(3):
                h, _ = np.histogram(arr[...,c], bins=bins, range=(0,255))
                hsum[c] += h
        hsum += 1e-8
        hsum = (hsum / hsum.sum(axis=1, keepdims=True))
        return hsum
    hA, hB = mean_hist(paths_a), mean_hist(paths_b)
    chi = 0.5 * np.sum((hA - hB)**2 / (hA + hB))
    return float(chi)

def compute_pad(feats_a, feats_b, C=1.0):
    """Proxy A-distance via domain classification accuracy."""
    X = np.vstack([feats_a, feats_b])
    y = np.hstack([np.zeros(len(feats_a)), np.ones(len(feats_b))])
    # standardize
    mu, std = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / std
    idx = np.arange(len(Xn)); np.random.shuffle(idx)
    split = int(0.7 * len(Xn))
    tr, te = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=2000, C=C, n_jobs=-1)
    clf.fit(Xn[tr], y[tr])
    acc = accuracy_score(y[te], clf.predict(Xn[te]))
    err = 1 - acc
    pad = 2 * (1 - 2*err)  # Ben-David proxy A-distance
    return float(pad), float(acc)

def compute_mmd_rbf(feats_a, feats_b, gammas=(1e-3, 1e-4, 1e-5)):
    """Multi-kernel RBF MMD^2 (unbiased)."""
    Xa, Xb = feats_a, feats_b
    def rbf(x, y, gamma):
        x_norm = (x**2).sum(1)[:,None]
        y_norm = (y**2).sum(1)[None,:]
        K = np.exp(-gamma * (x_norm + y_norm - 2 * x @ y.T))
        return K
    m, n = Xa.shape[0], Xb.shape[0]
    mmd2 = 0.0
    for g in gammas:
        Kaa = rbf(Xa, Xa, g); np.fill_diagonal(Kaa, 0)
        Kbb = rbf(Xb, Xb, g); np.fill_diagonal(Kbb, 0)
        Kab = rbf(Xa, Xb, g)
        mmd2 += Kaa.sum()/(m*(m-1)) + Kbb.sum()/(n*(n-1)) - 2*Kab.mean()
    return float(mmd2 / len(gammas))

def compute_linear_cka(X, Y):
    """
    Linear CKA between two matrices (n x d) of paired samples.
    Here we random-pair samples after z-scoring within each domain.
    """
    n = min(X.shape[0], Y.shape[0])
    X = X[:n]; Y = Y[:n]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    # Center rows
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, ord='fro')**2
    xkx = np.linalg.norm(Xc.T @ Xc, ord='fro')**2
    yky = np.linalg.norm(Yc.T @ Yc, ord='fro')**2
    return float(hsic / (math.sqrt(xkx) * math.sqrt(yky) + 1e-12))

def compute_lpips_set_distance(paths_a, paths_b, sample_pairs=500, device="cuda"):
    """
    Approximate set-to-set LPIPS: mean LPIPS over random A↔B pairs.
    """
    if not HAS_LPIPS:
        return None
    loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    nA, nB = len(paths_a), len(paths_b)
    if nA == 0 or nB == 0:
        return None
    sp = min(sample_pairs, nA*nB)
    vals = []
    with torch.no_grad():
        for _ in tqdm(range(sp), desc="LPIPS", leave=False):
            ia = random.randrange(nA); ib = random.randrange(nB)
            a = tf(pil_loader(paths_a[ia])).unsqueeze(0).to(device)
            b = tf(pil_loader(paths_b[ib])).unsqueeze(0).to(device)
            v = loss_fn(a*2-1, b*2-1).item()  # expect [-1,1] range
            vals.append(v)
    return float(np.mean(vals))

def save_embedding_plot(feats_a, feats_b, out_png, method="tsne", name_suffix=None):
    X = np.vstack([feats_a, feats_b])
    y = np.array([0]*len(feats_a) + [1]*len(feats_b))

    Xp = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)

    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=30, min_dist=0.1, metric="euclidean")
        Z = reducer.fit_transform(Xp)
    else:
        tsne_kwargs = dict(n_components=2, init="pca", random_state=42,
                           perplexity=min(30, max(5, (len(X)//50))))
        from packaging import version
        import sklearn
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000
        Z = TSNE(**tsne_kwargs).fit_transform(Xp)

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[y==0,0], Z[y==0,1], s=6, alpha=0.6, label="Domain A")
    plt.scatter(Z[y==1,0], Z[y==1,1], s=6, alpha=0.6, label="Domain B")
    plt.legend()
    plt.title(f"{method.upper()} of Inception Features")
    plt.tight_layout()

    if name_suffix:
        stem = out_png.stem + f"_{_slugify(name_suffix)}"
        out_png = out_png.with_name(f"{stem}{out_png.suffix}")

    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


# ---------------------------
# Main
# ---------------------------
def main(args):
    set_seed(42)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    paths_a = list_images(args.domain_a, args.max_images)
    paths_b = list_images(args.domain_b, args.max_images)
    if len(paths_a) == 0 or len(paths_b) == 0:
        raise RuntimeError("No images found. Check --domain_a/--domain_b and extensions.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Features (Inception)
    model = InceptionFeat().to(device)
    feats_a = load_in_batches(paths_a, model, batch_size=args.batch_size)
    feats_b = load_in_batches(paths_b, model, batch_size=args.batch_size)

    # FID (mean/cov)
    muA, muB = feats_a.mean(0), feats_b.mean(0)
    sigA = np.cov(feats_a, rowvar=False)
    sigB = np.cov(feats_b, rowvar=False)
    fid = compute_fid(muA, sigA, muB, sigB)

    # KID
    kid_mean, kid_std = compute_kid(feats_a, feats_b)

    # PAD (with logistic regression)
    pad, dom_acc = compute_pad(feats_a, feats_b, C=1.0)

    # MMD
    mmd2 = compute_mmd_rbf(feats_a, feats_b)

    # CKA (paired)
    cka = compute_linear_cka(feats_a, feats_b)

    # LPIPS (approx. set distance)
    lpips_val = None
    if args.lpips and HAS_LPIPS:
        lpips_val = compute_lpips_set_distance(paths_a, paths_b, sample_pairs=args.lpips_pairs, device=device)

    # Color histogram chi-square
    chi_rgb = compute_color_hist_distance(paths_a, paths_b, bins=64)

    # Embedding plots
    tsne_file = save_embedding_plot(feats_a, feats_b, out / "tsne.png",
                                method="tsne", name_suffix=args.eval_name)
    umap_file = None
    if HAS_UMAP:
        umap_file = save_embedding_plot(feats_a, feats_b, out / "umap.png",
                                        method="umap", name_suffix=args.eval_name)



    # ---- Optional: task-level metrics ----
    task_metrics = None
    if args.task:
        if not args.model_py:
            raise ValueError("--model_py is required when --task is set.")
        model, tform = _load_user_model(args.model_py, device)
        if args.task == "classification":
            if not args.labels_csv:
                raise ValueError("--labels_csv is required for classification.")
            task_metrics = _eval_classification(args.domain_a, args.labels_csv, model, tform, device,
                                                batch_size=args.batch_size)
        elif args.task == "segmentation":
            if (args.mask_dir is None) or (args.num_classes is None):
                raise ValueError("--mask_dir and --num_classes are required for segmentation.")
            task_metrics = _eval_segmentation(args.domain_a, args.mask_dir, model, tform, device,
                                            num_classes=args.num_classes, ignore_index=args.ignore_index,
                                            mask_suffix=args.mask_suffix, batch_size=args.batch_size)
        else:  # detection
            if not args.coco_gt:
                raise ValueError("--coco_gt is required for detection.")
            task_metrics = _eval_detection_coco(args.domain_a, args.coco_gt, model, tform, device,
                                                score_thresh=args.score_thresh, batch_size=1)



    # Save summary
    summary = {
        "evaluation_name": args.eval_name,
        "num_images": {"A": len(paths_a), "B": len(paths_b)},
        "features": "InceptionV3 pool3 (2048-D)",
        "metrics": {
            "FID": fid,
            "KID_mean": kid_mean,
            "KID_std": kid_std,
            "PAD": pad,
            "DomainClassifierAccuracy": dom_acc,
            "MMD2_RBF": mmd2,
            "CKA_linear_paired": cka,
            "LPIPS_set_to_set": lpips_val,
            "ChiSquare_RGB_Hist": chi_rgb
        },
        "plots": {
            "TSNE": str(tsne_file),
            "UMAP": (str(umap_file) if HAS_UMAP else None)
        },
        "task_metrics": task_metrics,
        "notes": {
            "FID": "Fréchet Inception Distance; lower is better. Measures distance between Gaussian fits to Inception features of the two domains. Sensitive to both mean and covariance shifts. Good for global distribution similarity.",
            "KID": "Kernel Inception Distance; mean and std shown. Polynomial kernel MMD on Inception features; unbiased and more stable on small datasets than FID. Lower is better.",
            "PAD": "Proxy A-distance; lower is better (0 = indistinguishable domains). Computed via domain classifier accuracy on Inception features. High PAD indicates large separability between domains.",
            "DomainClassifierAccuracy": "Accuracy of logistic regression domain classifier in PAD calculation. Higher accuracy indicates a larger domain gap. 0.5 means domains are indistinguishable.",
            "MMD2_RBF": "Squared Maximum Mean Discrepancy with multi-scale RBF kernels on Inception features; lower is better. Measures distance between domain feature distributions without assuming Gaussianity.",
            "CKA_linear_paired": "Linear Centered Kernel Alignment between paired domain features (paired by index after truncation to min(N_A,N_B)). Higher means more similar representations (1 = identical). Sensitive to feature-space geometry.",
            "LPIPS_set_to_set": "Learned Perceptual Image Patch Similarity averaged over random cross-domain image pairs. Lower is better; requires `lpips` package. Correlates with human perceptual similarity.",
            "ChiSquare_RGB_Hist": "Chi-square distance between average RGB histograms of the domains; lower indicates more similar global color distribution. Ignores spatial structure.",
            "TSNE": "2D t-SNE embedding of Inception features, saved as a PNG file (suffix includes eval_name if provided). Useful for visualizing local feature-space overlap between domains.",
            "UMAP": "2D UMAP embedding of Inception features (if available), saved as a PNG file (suffix includes eval_name if provided). Preserves more global structure than t-SNE.",
            "Features": "Unless task metrics are requested, all metrics are computed using InceptionV3 pool3 (2048-D) features extracted from resized and center-cropped RGB images.",
            "Limitations": "Feature-level metrics capture distribution similarity, not task-specific performance. Small domain gap in these metrics does not guarantee equal downstream accuracy.",
            "TaskMetrics_Classification": "If computed, includes accuracy, per-class precision/recall/F1, and confusion matrix on labeled classification data using a user-provided model.",
            "TaskMetrics_Segmentation": "If computed, includes mean IoU, per-class IoU, and pixel accuracy on segmentation masks using a user-provided model.",
            "TaskMetrics_Detection": "If computed, includes COCO mAP/mAR metrics on detection datasets using a user-provided model. Evaluated via pycocotools.",
            "GeneralAdvice": "Always interpret metrics in context of the downstream task. A combination of feature-level and task-specific metrics gives a more reliable picture of the domain gap."
        }
    }

    json_name = "summary.json"
    if args.eval_name:
        json_name = f"summary_{_slugify(args.eval_name)}.json"
    if args.out_json_suffix:
        json_name = json_name.replace(".json", f"_{_slugify(args.out_json_suffix)}.json")


    with open(out / json_name, "w") as f:
        json.dump(summary, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain gap evaluation for two image folders.")
    parser.add_argument("--domain_a", type=str, required=True)
    parser.add_argument("--domain_b", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./domain_gap_report")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None, help="cap number of images per domain")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--lpips", action="store_true", help="compute LPIPS set distance (requires `lpips`)")
    parser.add_argument("--lpips_pairs", type=int, default=500)
    parser.add_argument("--eval_name", type=str, default=None, help="name for the evaluation run")
    # ---- Task metrics (optional) ----
    parser.add_argument("--task", choices=["classification", "segmentation", "detection"],
                        help="If set, compute task-level metrics and append to summary.json")
    parser.add_argument("--model_py", type=str,
                        help="Path to model_def.py that provides load_model(device)->(model, transform)")
    parser.add_argument("--out_json_suffix", type=str, default=None,
                        help="Optional extra suffix for the summary filename")

    # classification
    parser.add_argument("--labels_csv", type=str, help="CSV with header filename,label")

    # segmentation
    parser.add_argument("--mask_dir", type=str, help="Folder with GT masks")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--ignore_index", type=int, default=None)
    parser.add_argument("--mask_suffix", type=str, default=".png")

    # detection
    parser.add_argument("--coco_gt", type=str, help="COCO GT JSON for detection")
    parser.add_argument("--score_thresh", type=float, default=0.0)

    args = parser.parse_args()
    main(args)


# Example usage
# Classification:
# python evaluate_domain_gap.py \
#   --domain_a /data/classif/images \
#   --domain_b /data/classif/images_target \
#   --output_dir ./reports \
#   --eval_name run_cls \
#   --task classification \
#   --model_py model_def.py \
#   --labels_csv /data/classif/labels_val.csv

# Segmentation:
# python evaluate_domain_gap.py \
#   --domain_a /data/seg/images_val \
#   --domain_b /data/seg/images_target \
#   --output_dir ./reports \
#   --eval_name run_seg \
#   --task segmentation \
#   --model_py model_def.py \
#   --mask_dir /data/seg/masks_val \
#   --num_classes 7 \
#   --ignore_index 255

# Detection:
# python evaluate_domain_gap.py \
#   --domain_a /data/det/images_val \
#   --domain_b /data/det/images_target \
#   --output_dir ./reports \
#   --eval_name run_det \
#   --task detection \
#   --model_py model_def.py \
#   --coco_gt /data/det/instances_val.json \
#   --score_thresh 0.05
