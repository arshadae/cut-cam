#!/usr/bin/env python3
"""
cam_infer.py
Run inference with CUTCAM and save:
 - translated image (fake_B)
 - CAM maps for input and output
 - overlays (heatmap blended on top of images)

Assumes repo structure similar to CUT/CycleGAN:
 - options/ (TestOptions)
 - models/ (CUTCAMModel)
 - data/ (datasets)
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt  # used to create heatmap images

import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def to_numpy_uint8(img_t, scale_0_1=False):
    """
    img_t: torch.Tensor shape [B,C,H,W] or [C,H,W], values in [-1,1] (typical CUT)
    Returns np.uint8 (H,W) or (H,W,3)
    """
    if img_t.dim() == 4:
        img_t = img_t[0]
    x = img_t.detach().cpu().float()
    if x.shape[0] == 1:
        x = x[0:1, ...]  # (1,H,W)
    # [-1,1] -> [0,1]
    x = (x * 0.5) + 0.5
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x[0].numpy()  # (H,W)
        x = (x * 255.0).round().astype(np.uint8)
    else:
        x = x.permute(1, 2, 0).numpy()  # (H,W,C)
        x = (x * 255.0).round().astype(np.uint8)
    return x


def heatmap_to_rgba(cam_01, cmap_name="jet", alpha=0.5):
    """
    cam_01: np.ndarray in [0,1], shape (H,W).
    Returns uint8 RGBA heatmap image (H,W,4), colormap applied, with alpha premultiplied.
    """
    cm = plt.get_cmap(cmap_name)
    rgba = cm(cam_01)  # (H,W,4) in [0,1]
    rgba[..., 3] = alpha
    rgba = (rgba * 255).astype(np.uint8)
    return rgba


def overlay_heatmap_on_image(img_uint8, heatmap_rgba):
    """
    img_uint8: grayscale or RGB uint8 (H,W) or (H,W,3)
    heatmap_rgba: (H,W,4) uint8 with alpha channel
    Returns blended uint8 RGB image (H,W,3)
    """
    if img_uint8.ndim == 2:
        base_rgb = np.stack([img_uint8]*3, axis=-1)
    else:
        base_rgb = img_uint8.copy()

    # normalize to [0,1]
    base = base_rgb.astype(np.float32) / 255.0
    hm = heatmap_rgba.astype(np.float32) / 255.0
    alpha = hm[..., 3:4]  # (H,W,1)
    hm_rgb = hm[..., :3]

    out = (1 - alpha) * base + alpha * hm_rgb
    out = (out * 255.0).round().astype(np.uint8)
    return out


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    # Parse options as in CUT/CycleGAN test
    opt = TestOptions().parse()  # this will read CLI args
    # typical test-time defaults
    opt.num_threads = 0
    opt.batch_size = 1  # MUST be 1 for per-image saving
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.eval = True

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # resolve CAM layer id from index (see model refactor)
    nce_layers = [int(i) for i in opt.nce_layers.split(',')]
    cam_index = getattr(opt, 'cam_layer', -1)
    cam_layer_id = nce_layers[cam_index] if cam_index != -1 else nce_layers[-1]

    # output dir
    save_dir = Path(opt.results_dir) / opt.name / str(opt.epoch)
    ensure_dir(save_dir)

    print(f"[INFO] Using CAM layer id: {cam_layer_id} from nce_layers {nce_layers}")
    print(f"[INFO] Saving to: {save_dir}")

    # run
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, total=min(len(dataset), opt.num_test) if opt.num_test > 0 else len(dataset))):
            if opt.num_test > 0 and i >= opt.num_test:
                break

            model.set_input(data)        # unpack data
            model.test()                 # forward: sets model.fake_B

            # tensors
            real_A = model.real_A        # source
            fake_B = model.fake_B        # translated
            # Optional: if dataset provides labels, they’re in data.get('cls'), but not required for CAM at test.

            # compute CAMs (uses the built-in helper from your model)
            cam_in  = model._compute_cam_map(real_A, use_input=True)   # [B,1,H,W], [0,1]
            cam_out = model._compute_cam_map(fake_B, use_input=False)  # [B,1,H,W], [0,1]

            # convert to uint8 for saving
            real_A_np = to_numpy_uint8(real_A)        # (H,W) or (H,W,3)
            fake_B_np = to_numpy_uint8(fake_B)

            cam_in_np  = (cam_in[0,0].cpu().numpy()  * 255.0).round().astype(np.uint8)
            cam_out_np = (cam_out[0,0].cpu().numpy() * 255.0).round().astype(np.uint8)

            # overlays
            hm_in_rgba  = heatmap_to_rgba(cam_in_np / 255.0, cmap_name="jet", alpha=0.5)
            hm_out_rgba = heatmap_to_rgba(cam_out_np / 255.0, cmap_name="jet", alpha=0.5)

            overlay_in  = overlay_heatmap_on_image(real_A_np, hm_in_rgba)
            overlay_out = overlay_heatmap_on_image(fake_B_np, hm_out_rgba)

            # paths
            img_name = Path(model.image_paths[0]).stem if hasattr(model, 'image_paths') else f"img_{i:06d}"
            p_fake   = save_dir / f"{img_name}_fake.png"
            p_cam_i  = save_dir / f"{img_name}_cam_input.png"
            p_cam_o  = save_dir / f"{img_name}_cam_output.png"
            p_ov_i   = save_dir / f"{img_name}_overlay_input.png"
            p_ov_o   = save_dir / f"{img_name}_overlay_output.png"

            # save
            Image.fromarray(fake_B_np if fake_B_np.ndim == 3 else fake_B_np).save(p_fake)
            Image.fromarray(cam_in_np).save(p_cam_i)
            Image.fromarray(cam_out_np).save(p_cam_o)
            Image.fromarray(overlay_in).save(p_ov_i)
            Image.fromarray(overlay_out).save(p_ov_o)

    print("[DONE] Inference & CAM overlays saved.")


if __name__ == "__main__":
    main()
