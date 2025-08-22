# CUTCAM: Contrastive Unpaired Translation with Class Activation Maps

**CUTCAM** is a single-stage, unpaired, **explainability-guided** GAN for image translation.  
It extends **CUT (Contrastive Unpaired Translation, ECCV 2020)** by integrating **Class Activation Maps (GradCAM)** directly into the generator, making translations **semantically faithful, class-attentive, and explainable**.

> Works for MWIR and beyond (SAR, medical, etc.) — not tied to any single modality.

---

## ✨ Key Ideas

- **One translator** (G) + **one PatchGAN** (D): unpaired training, single stage  
- **CUT/PatchNCE** keeps **content**; GradCAM keeps **semantics**  
- CAMs are computed **from G’s own encoder features** via a tiny **aux head** (no frozen ResNet18 needed)

---


## 📖 Abstract

High-fidelity **Mid-Wave Infrared (MWIR)** imagery is critical in surveillance, defense, and machine vision. However, acquiring real annotated MWIR data is expensive and limited. While physics-based simulators can generate synthetic MWIR, the domain gap (texture mismatch, semantic drift) reduces their downstream utility.

CUTCAM bridges this gap by combining:
- **CUT/PatchNCE** → preserves content without paired supervision.  
- **Auxiliary classifier head + GradCAM** → enforces semantic consistency during translation.  
- **Explainability maps** → visualize where the generator “looks” when retaining class-critical features.

The result: **realistic, semantically aligned translations** that outperform vanilla CUT and Pix2Pix on realism (FID) and semantic preservation (SSIM, CC).

---

## 🚀 Contributions

1. **Explainability-Guided Translation**  
   Introduce GradCAM invariance loss inside the generator, aligning semantic attention maps between input and translated images.

2. **Auxiliary Classifier Head**  
   Build a lightweight classifier on G’s encoder to compute logits + CAMs — removing the need for an external pretrained ResNet.

3. **Class-Consistency Loss**  
   Enforce stable classification before and after translation, ensuring outputs remain task-relevant.

4. **CAM-Weighted Adversarial Training**  
   Discriminator focuses more on semantically important regions (weighted by CAM maps).

5. **Identity & Edge Losses (optional)**  
   Preserve tones, gradients, and small hot targets common in MWIR.

6. **Explainability Out-of-the-Box**  
   Inference saves both translated images and **GradCAM overlays** for transparency, just like Fig. 4 in our paper.

---


## 🛠️ Installation

```bash
# Python 3.9+ recommended
git clone <your-repo-url>.git
cd <your-repo-folder>

# (recommended) Either use container or create a venv/conda env
pip install -r requirements.txt
```

## Minimal Requirements

`Python 3.8+`
`PyTorch >= 1.9`
`torchvision`
`numpy, pillow, tqdm, matplotlib`
(optional) `opencv-python`


## Data

```
datasets/
  your_data/
    trainA/   # source domain (e.g., simulator or domain A)
    trainB/   # target domain (e.g., real or domain B)
    testA/
    testB/
    
```
If you have labels for domain A samples, add them to your dataset (int64 class id).  
The model will auto‑fallback to pseudo‑labels when labels are absent.  

## Training (examples)

```bash
python train.py \
  --dataroot datasets/your_data \
  --name cutcam_run \
  --model cutcam \
  --CUT_mode CUTCAM \
  --nce_layers 0,4,8,12,16 \
  --cam_layer -1 \
  --num_classes 7 \
  --lambda_GAN 1 --lambda_NCE 1 \
  --lambda_CLS 1 --lambda_CAM 25 \
  --lambda_IDT 10 --nce_idt \
  --lambda_EDGE 5 \
  --cam_weight_adv \
  --netG resnet_9blocks --netD basic \
  --input_nc 3 --output_nc 3 \
  --batch_size 4 --preprocess resize_and_crop --load_size 286 --crop_size 256

```

Notes: Read options directory for arugments

## Inference + CAM Overlays
Use the included `cam_infer.py` to translate and save:  
- translated images (G(A) → B̂)
- CAM maps for input and output
- overlays (heatmap on top of image)

```bash
python cam_infer.py \
  --dataroot datasets/your_data \
  --name cutcam_run \
  --model cutcam \
  --CUT_mode CUTCAM \
  --phase test \
  --epoch latest \
  --results_dir results/cam_vis \
  --nce_layers 0,4,8,12,16 \
  --cam_layer -1 \
  --num_test 10000 \
  --input_nc 3 --output_nc 3
```
Outputs (per image) in results/cam_vis/cutcam_run/latest/:  
- `*_fake.png` — translated image
- `*_cam_input.png` — CAM(x)
- `*_cam_output.png` — CAM(G(x))
- `*_overlay_input.png` — heatmap over input
- `*_overlay_output.png` — heatmap over output

## Important Options

|Option         |                 What it does                                  |
|-------------- |---------------------------------------------------------------|
|--nce_layers   |Feature layers used by PatchNCE and selectable for CAM         |
|--cam_layer    |Index into nce_layers for CAM (e.g., -1 → last entry)          |
|--lambda_CAM	|Weight for CAM invariance loss                                 |
|--lambda_CLS	|Weight for classification consistency                          |
|--lambda_IDT	|Identity loss weight on real B (stabilizes tones)              |
|--lambda_EDGE	|Edge loss (Sobel) to preserve small hot targets                |
|--cam_weight_adv	|Weight PatchGAN loss by CAM importance                     |

Internally, the code resolves `cam_layer` index to the real layer id (e.g., 2 → layer 8 if nce_layers=0,4,8,12,16).

## 📊 Results (Example from MWIR) [to be updated]

- FID reduced by 5–6% vs. CUT alone.
- SSIM improved from 0.776 → 0.808.
- CAM overlays show generator progressively focuses on class-critical regions (e.g., MWIR targets).


## 🖼️ Example Outputs [to be upated]

**Reproducing Fig. 4‑style visuals from the paper**
Run cam_infer.py (provided). It saves plain CAMs + overlays. For epoch‑wise grids, you can:  
- run inference with multiple epochs and stitch images, or
- log CAMs during training and save a per‑epoch panel.

## 📝 Citations
If you use this repo, please cite:  
```bibtex
@inproceedings{arshad2025cutcam,
  title     = {CUTCAM: Explainability-Guided Contrastive Unpaired Translation with Class Activation Maps},
  author    = {Muhammad Awais Arshad and Hyochoong Bang},
  booktitle = {GitHub Repo},
  year      = {2025}
}
```
Also cite CUT
```bibtex
@inproceedings{park2020cut,
  title     = {Contrastive Learning for Unpaired Image-to-Image Translation},
  author    = {Park, Taesung and Efros, Alexei A. and Zhang, Richard and Zhu, Jun-Yan},
  booktitle = {ECCV},
  year      = {2020}
}
```

## 🙏 Acknowledgements

- [Taesung Park et al., CUT (ECCV 2020)](https://github.com/taesungp/contrastive-unpaired-translation)
- [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Our code is built on top of these excellent open-source repos.

## License
MIT License (follow upstream CUT). See `LICENSE`



## Planned Evalutions for unpaired I2I techniques [Has to be completed yet]

### Evaluation Metrics
Besides FID, we plan following evaluations:

- **Inception Score (IS)**
- **Kernel Inception Distance (KID)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**
- **Precision and Recall for Generative Models**
- **Classification Accuracy with a pretrained model**
- **Semantic Segmentation metrics (e.g., mIoU)**
- **User Study / Human Evaluation**

These metrics help assess image quality, diversity, perceptual similarity, and semantic consistency.

### Visualization Techniques
To qualitatively validate the results, we also consider following visualization techniques

- **Side-by-side comparisons** of input, generated, and target images
- **t-SNE or UMAP plots** of feature embeddings for real and generated images
- **Difference maps** to highlight changes
- **Activation or attention maps** (e.g., Grad-CAM)
- **Montages or grids** of generated samples
- **Histogram plots** for color or feature distributions
- **Semantic segmentation overlays** for
