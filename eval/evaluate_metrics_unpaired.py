import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---- 0. Target Directories ---- #
sim = './results/oktal_target/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_B' # sim (oktal)
real = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_B' # real (dsiac)
add = './results/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_A'
psim = './results/psim_target/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/real_B' # psim (from real using cut)
psim_cyclegan = './third_party/pytorch-CycleGAN-and-pix2pix/results/mwir_cyclegan/eval/fake_B/' # psim-cyclegan (from real using cyclegan)

psim2preal_p2p = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_0.0001/test_latest/images/fake_B/' # psim2preal-p2p (from psim-cut using Pix2Pix)
psim2preal_p2pgm = './results/psim_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/fake_B/' # psim2preal-p2pgm (from psim-cut using Pix2Pix)
sim2preal_p2p = './results/oktal_start/pix2pix_gradcam_psim2preal_1_10_0.0001/test_latest/images/fake_B' # sim2preal-p2p
sim2preal_p2pgm = './results/oktal_start/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/fake_B' # sim2preal-p2pgm
add2preal_p2pgm = './results/pix2pix_gradcam_psim2preal_1_10_50/test_latest/images/fake_B'


ResNet18_Path = './checkpoints/classifiers/resnet18_mwir.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 1. Feature Extractor ---- #
class FeatureExtractor(nn.Module):
    def __init__(self, layers=3):
        super().__init__()

        # Load custom saved ResNet18
        resnet = models.resnet18(weights=None) # 7 = number of classes in saved model
        resnet.fc = nn.Linear(resnet.fc.in_features,7)
        state_dict = torch.load(ResNet18_Path, map_location=device)
        resnet.load_state_dict(state_dict)
        resnet.eval()

        # Load default pretrained ResNet18
        # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Extract first 'layers' blocks (focus on low-level modality)
        self.features = nn.Sequential(*list(resnet.children())[:layers])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

# ---- 2. Dataset Loader ---- #
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# ---- 3. MMD Computation ---- #
def compute_mmd(x, y, kernel='rbf', sigma=1.0, batch_size=256):
    def rbf_kernel(a, b, sigma):
        dist = torch.cdist(a, b) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    # x = x[:batch_size]  # Optional: subsample
    # y = y[:batch_size]
    return rbf_kernel(x, x, sigma).mean() + rbf_kernel(y, y, sigma).mean() - 2 * rbf_kernel(x, y, sigma).mean()

# ---- 4. CORAL Computation ---- #
def compute_coral(source, target):
    """ source, target: [N, D] tensors (max ~500 x 512) """
    source_c = source - source.mean(0)
    target_c = target - target.mean(0)

    cs = (source_c.T @ source_c) / (source_c.size(0) - 1)
    ct = (target_c.T @ target_c) / (target_c.size(0) - 1)

    return ((cs - ct) ** 2).sum() / (4 * source.size(1) ** 2)

# ---- 5. Feature Tensor Loader ---- #
def feature_loader(folder_a, batch_size=16, layer_depth=3, max_samples=1000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    loader_a = DataLoader(ImageFolderDataset(folder_a, transform), batch_size=batch_size, shuffle=True)

    model = FeatureExtractor(layer_depth).eval().cuda()

    features_a = []

    with torch.no_grad():
        for imgs in loader_a:
            feats = model(imgs.cuda())
            feats = nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1).cpu()

            features_a.append(feats)
            if sum([f.shape[0] for f in features_a]) >= max_samples:
                break

    F_a = torch.cat(features_a, dim=0)[:max_samples]
    return F_a

# ---- 6. Feature Visualizer ---- #
def visualize_features_multi(features, labels, label_names, method='tsne', perplexity=30, save_prefix='modalities'):
    """
    Visualize high-dimensional features using PCA or t-SNE.

    Args:
        features: (N, D) NumPy array of feature vectors.
        labels:   (N,) NumPy array of integer labels (e.g., 0, 1, 2, ...).
        label_names: list of string names corresponding to labels.
        method: 'pca' or 'tsne'.
        perplexity: t-SNE perplexity parameter (ignored for PCA).
        save_prefix: base filename prefix for saving the plot.
    """
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(features)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(features)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    # Color palette (20-class safe)
    cmap = plt.get_cmap('tab20')
    num_classes = len(label_names)

    # Plotting
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        pts = reduced[labels == i]
        # plt.scatter(pts[:, 0], pts[:, 1], label=label_names[i], color=cmap(i % 20), alpha=0.6, s=35)
        plt.scatter(pts[:, 0], pts[:, 1], label=label_names[i], alpha=0.6, s=35)

    plt.title(f"{method.upper()} Visualization of Modalities")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)

    # Clean filename
    label_str = '_'.join(l.replace(" ", "_").lower() for l in label_names)
    fname_base = f"{save_prefix}_{method.lower()}_{label_str}"

    plt.tight_layout()
    plt.savefig(f"{fname_base}.png", dpi=600)
    # plt.savefig(f"{fname_base}.pdf")
    print(f"✅ Saved plots as {fname_base}.png and .pdf")
    plt.close()

# ---- 7. Example Usage ---- #
if __name__ == '__main__':

    feat_sim = feature_loader(sim, batch_size=16, layer_depth=3, max_samples=1000)
    feat_real = feature_loader(real, batch_size=16, layer_depth=3, max_samples=1000)

    feat_psim = feature_loader(psim, batch_size=16, layer_depth=3, max_samples=1000)
    feat_psim_cyclegan = feature_loader(psim_cyclegan, batch_size=16, layer_depth=3, max_samples=1000)

    feat_psim2preal_p2p = feature_loader(psim2preal_p2p, batch_size=16, layer_depth=3, max_samples=1000)
    feat_psim2preal_p2pgm = feature_loader(psim2preal_p2pgm, batch_size=16, layer_depth=3, max_samples=1000)

    feat_sim2preal_p2p = feature_loader(sim2preal_p2p, batch_size=16, layer_depth=3, max_samples=1000)
    feat_sim2preal_p2pgm = feature_loader(sim2preal_p2pgm, batch_size=16, layer_depth=3, max_samples=1000)

    feat_add = feature_loader(add, batch_size=16, layer_depth=3, max_samples=1000)
    feat_add2preal_p2pgm = feature_loader(add2preal_p2pgm, batch_size=16, layer_depth=3, max_samples=1000)
    

    # print("Computing Modality Gap b/w real and sim ...")
    # mmd = compute_mmd(X, Y)
    # coral = compute_coral(X, Y)
    # modality_gap = mmd + 1.0 * coral
    # print(f"Modality Gap: {modality_gap:.4f} | MMD: {mmd:.4f} | CORAL: {coral:.4f}")

    # print("Computing Modality Gap b/w sim and psim-cut ...")
    # mmd = compute_mmd(X, Z1)
    # coral = compute_coral(X, Z1)
    # modality_gap = mmd + 1.0 * coral
    # print(f"Modality Gap: {modality_gap:.4f} | MMD: {mmd:.4f} | CORAL: {coral:.4f}")

    # print("Computing Modality Gap b/w real and psim-cut ...")
    # mmd = compute_mmd(Y, Z1)
    # coral = compute_coral(Y, Z1)
    # modality_gap = mmd + 1.0 * coral
    # print(f"Modality Gap: {modality_gap:.4f} | MMD: {mmd:.4f} | CORAL: {coral:.4f}")


    # Assuming X, Y, Z are feature tensors for the target modalities
    f1_np = feat_sim.cpu().numpy() # sim
    f2_np = feat_real.cpu().numpy() # real
    f3_np = feat_psim.cpu().numpy() # psim
    f4_np = feat_psim_cyclegan.cpu().numpy() # psim-cyclegan
    f5_np = feat_psim2preal_p2p.cpu().numpy() # psim2peal-p2p
    f6_np = feat_psim2preal_p2pgm.cpu().numpy() # psim2preal-p2pgm
    f7_np = feat_sim2preal_p2p.cpu().numpy() # sim2preal-p2p
    f8_np = feat_sim2preal_p2pgm.cpu().numpy() # sim2preal-p2pgm
    f9_np = feat_add.cpu().numpy() # ADD synthetic data
    f10_np = feat_add2preal_p2pgm.cpu().numpy() # add2preal-p2pgm


    # # visualize all modalities
    # features = np.vstack([f1_np, f2_np, f3_np, f4_np, f5_np, f6_np, f7_np, f8_np])
    # modality_labels = np.array([0]*len(f1_np) + [1]*len(f2_np) + [2]*len(f3_np) + [3]*len(f4_np) + [4]*len(f5_np) + [5]*len(f6_np) + [6]*len(f7_np) + [7]*len(f8_np))
    # label_names = ['sim', 'real', 'psim-cut', 'psim-cyclegan', 'psim2preal-p2p', 'psim2preal-p2pgm', 'sim2preal-p2p', 'sim2preal-p2pgm']



    # # visualize selective modalities
    # features = np.vstack([f1_np, f2_np, f5_np, f6_np])
    # modality_labels = np.array([0]*len(f1_np) + [1]*len(f2_np) + [2]*len(f5_np) + [3]*len(f6_np))
    # label_names = ['sim', 'real', 'psim2preal-p2p', 'psim2preal-p2pgm']


    # visualize selective modalities
    # features = np.vstack([f1_np, f2_np, f7_np, f8_np])
    # modality_labels = np.array([0]*len(f1_np) + [1]*len(f2_np) + [2]*len(f7_np) + [3]*len(f8_np))
    # label_names = ['sim', 'real', 'sim2preal-p2p', 'sim2preal-p2pgm']


    features = np.vstack([f2_np, f9_np, f10_np])
    modality_labels = np.array([0]*len(f2_np) + [1]*len(f9_np) + [2]*len(f10_np))
    label_names = ['real', 'add-synth', 'add2preal-p2pgm']



    visualize_features_multi(features, modality_labels, label_names, method='tsne') 
    visualize_features_multi(features, modality_labels, label_names, method='pca') 
