# CUT-CAM: Contrastive Unpaired Translation with Class Attentive Maps
### This work is a continuation from original work on [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) by Taesung Park. We refer our readers to the original [README](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/README.md) file for the context.




# CUTCAM
1. Perform thorough evalutions on Contrastive Unpaired Translation (CUT) and other unpaired image to image (I2I) translation techniques.
2. Evaluate existing I2I techniques  and find performance gaps.
3. Try to cover performance gaps in existing methods.

## 1. Planned Additional Evalutions for CUT

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






# Acknowledgements
- We borrowed heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) to develop our codebase. 
- We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation,  [drn](https://github.com/fyu/drn) for mIoU computation, and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch/) for the PyTorch implementation of StyleGAN2 used in our single-image translation setting.
- We also thank the teams providing orignal datasets (cityscapes, facades, maps, etc.) for training.
