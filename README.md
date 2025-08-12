# Bridging the Domain Gap: Beyond Pixel Translation and Feature Alignment
#### This work is a continuation from original work on [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) by Taesung Park. We refer our readers to the original [README](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/README.md) file for the context.



### Motivation
In many machine learning tasks, models trained on one domain (source) fail to generalize to another domain (target) due to domain shift—differences in data distribution.
This domain gap appears in:

- Low-level appearance: color, texture, illumination.
- Mid-level structure: object shapes, edge patterns.
- High-level semantics: scene composition, object co-occurrence.
- Feature space: differences in neural network embeddings.

While **image-to-image translation** (e.g., Pix2Pix, CycleGAN, CUT) and **feature-level domain adaptation** (e.g., DANN, CyCADA, DA-GAN) have improved cross-domain transfer, **the gap is rarely eliminated**.

### Why Existing Methods Fall Short

**Pixel-Level Translation Limitations**    
- Primarily matches style but not structure or semantics.  
- Cycle-consistency and identity losses prevent large geometric/semantic adjustments.  
- Introduces artifacts that form a new synthetic gap.  

**Feature-Level Adaptation Limitations**  
- Aligns global distributions, but class-conditional mismatches remain.  
- Can be fooled by adversarial objectives without real semantic alignment.  
- Risk of negative transfer if alignment destroys task-relevant domain cues.  
### Consequences of Incomplete Adaptation  
- Residual domain gap reduces accuracy on target data.  
- Misaligned class clusters cause errors in classification or segmentation.  
- Models overfit to synthetic styles, performing poorly on real-world variations.  
### Our Approach  

We aim to push beyond the limits of current methods by addressing both pixel-space and feature-space shortcomings. Planned improvements include:  

**Class-Conditional Feature Alignment**  
- Align features per class to avoid class confusion across domains.  
- Methods: Conditional Adversarial Domain Adaptation (CDAN), classifier discrepancy minimization.  

**Semantic Consistency in Translation**  
- Enforce that translated images retain task-specific semantics.  
- Example: segmentation maps or keypoints remain consistent after translation.  

**Joint Training of Translation and Task Models**  
- Optimize translation for downstream accuracy, not just visual realism.  

**Multi-Level Alignment**  
- Match distributions at multiple feature extractor layers (low-, mid-, high-level).  

**Target Domain Augmentation**  
- Increase coverage of rare or extreme variations in the target domain.  

**Domain Generalization**   
- Build features robust enough to handle unseen domains beyond the current target.  

### Expected Impact
This work will:
- Reduce residual domain gaps in both appearance and representation space.
- Improve class-conditional alignment for higher task accuracy.
- Deliver more robust models that generalize better to real-world scenarios.

### References
- Isola et al., Pix2Pix (2017)
- Zhu et al., CycleGAN (2017)
- Park et al., CUT (2020)
- Ganin et al., DANN (2016)
- Hoffman et al., CyCADA (2018)
- Tzeng et al., Adversarial Discriminative Domain Adaptation (2017)

If you’re reading this code, you’re looking at an ongoing effort to combine pixel translation, feature alignment, and semantic constraints into a unified framework for stronger, more general domain adaptation.





















## Additional Tasks
1. Perform thorough evalutions on Contrastive Unpaired Translation (CUT) and other unpaired image to image (I2I) translation techniques.
2. Evaluate existing I2I techniques  and find performance gaps.
3. Try to cover performance gaps in existing methods.

## 1. Planned Evalutions for unpaired I2I techniques

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
