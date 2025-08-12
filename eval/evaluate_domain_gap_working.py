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
        "notes": {
            "FID": "Fréchet Inception Distance; lower is better. Measures distance between Gaussian fits to Inception features of the two domains. Sensitive to both mean and covariance shifts.",
            "KID": "Kernel Inception Distance; mean and std shown. Polynomial kernel MMD on Inception features; unbiased and more stable on small datasets than FID.",
            "PAD": "Proxy A-distance; lower is better (0 = indistinguishable domains). Computed via domain classifier accuracy on Inception features.",
            "DomainClassifierAccuracy": "Accuracy of logistic regression domain classifier in PAD calculation. Higher accuracy indicates a larger domain gap.",
            "MMD2_RBF": "Squared Maximum Mean Discrepancy with multi-scale RBF kernels on Inception features; lower is better.",
            "CKA_linear_paired": "Linear Centered Kernel Alignment between paired domain features (paired by index after truncation to min(N_A,N_B)). Higher means more similar representations (1 = identical).",
            "LPIPS_set_to_set": "Learned Perceptual Image Patch Similarity averaged over random cross-domain image pairs. Lower is better; requires `lpips` package.",
            "ChiSquare_RGB_Hist": "Chi-square distance between average RGB histograms of the domains; lower indicates more similar color distribution.",
            "TSNE": "2D t-SNE embedding of Inception features, saved as a PNG file. Useful for visualizing feature-space overlap between domains.",
            "UMAP": "2D UMAP embedding of Inception features (if available), saved as a PNG file. Similar purpose as t-SNE but preserves more global structure.",
            "Features": "All metrics are computed using InceptionV3 pool3 (2048-D) features extracted from resized and center-cropped RGB images.",
            "Limitations": "Metrics capture feature and distribution similarity, not task-specific performance. A small domain gap here does not guarantee identical downstream accuracy."
        }
    }

    json_name = "summary.json"
    if args.eval_name:
        json_name = f"summary_{_slugify(args.eval_name)}.json"

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
    args = parser.parse_args()
    main(args)


# Example usage:
# python evaluate_domain_gap.py --domain_a path/to/domain/A --domain_b path/to/domain/B \
#   --output_dir ./domain_gap_report --batch_size 32 --max_images 5000 --eval_name "MyDomainEval"
