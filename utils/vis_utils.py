# vis_utils.py
# ──────────────────────────────────────────────────────────────────────────
#  Utility functions for saving:
#    • a single 3‑row composite (save_imgs)
#    • only the predicted mask (save_predicted_mask)
#    • three separate files (save_all_individually)
#
#  Works with tensors produced by PyTorch loaders or NumPy arrays.
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def _ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _as_rgb_numpy(img_t: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a tensor/array shaped (1,C,H,W) or (C,H,W) to H×W×C float [0,1].
    Handles both raw uint8 [0,255] and normalised ranges.
    """
    if isinstance(img_t, torch.Tensor):
        if img_t.ndim == 4:
            img_t = img_t.squeeze(0)
        img_np = img_t.detach().cpu().permute(1, 2, 0).float().numpy()
    else:  # numpy already?
        img_np = img_t
    if img_np.max() > 1.1:               # uint8 ➜ float
        img_np = img_np / 255.0
    elif img_np.min() < 0.0:             # [-1,1] ➜ [0,1]
        img_np = (img_np + 1.0) * 0.5
    return np.clip(img_np, 0.0, 1.0)


def _as_mask_numpy(msk_t: torch.Tensor | np.ndarray,
                   thresh: float = 0.5) -> np.ndarray:
    """
    Convert mask tensor/array to 2‑D uint8 {0,1}.
    Accepts (1,H,W), (H,W) or (1,1,H,W).
    """
    if isinstance(msk_t, torch.Tensor):
        if msk_t.ndim == 4:
            msk_t = msk_t.squeeze(0)
        if msk_t.ndim == 3:
            msk_t = msk_t.squeeze(0)
        msk_np = msk_t.detach().cpu().numpy()
    else:
        msk_np = msk_t
    return (msk_np > thresh).astype(np.uint8)

def save_imgs(
    img,
    msk,
    msk_pred,
    i: int,
    save_path: str,
    datasets: str,
    threshold: float = 0.5,
    test_data_name: str | None = None,
):
    """
    Save a single composite figure with 3 rows:
      1) input image
      2) ground‑truth mask
      3) predicted mask

    Args
    ----
    img, msk, msk_pred : tensor/ndarray pairs as produced in your loop
    i                  : index for filename
    save_path          : directory (created if missing)
    datasets           : dataset type; 'retinal' keeps raw values
    threshold          : prediction binarisation threshold
    test_data_name     : optional prefix in file name
    """
    _ensure_dir(save_path)
    prefix = f"{test_data_name}_" if test_data_name else ""
    fname  = os.path.join(save_path, f"{prefix}{i}.png")

    # to numpy
    img_np = _as_rgb_numpy(img)
    if datasets.lower() == "retinal":
        msk_np      = np.squeeze(msk,      axis=0)
        msk_pred_np = np.squeeze(msk_pred, axis=0)
    else:
        msk_np      = _as_mask_numpy(msk,      thresh=0.5)
        msk_pred_np = _as_mask_numpy(msk_pred, thresh=threshold)

    # plot
    plt.figure(figsize=(7, 15))
    plt.subplot(3, 1, 1); plt.imshow(img_np);      plt.axis("off")
    plt.subplot(3, 1, 2); plt.imshow(msk_np, cmap="gray");      plt.axis("off")
    plt.subplot(3, 1, 3); plt.imshow(msk_pred_np, cmap="gray"); plt.axis("off")
    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_predicted_mask(
    msk_pred,
    i: int,
    save_path: str,
    datasets: str,
    threshold: float = 0.5,
    test_data_name: str | None = None,
):
    """Save only the predicted binary mask as PNG."""
    _ensure_dir(save_path)
    prefix = f"{test_data_name}_" if test_data_name else ""
    fname  = os.path.join(save_path, f"{prefix}{i}_pred.png")

    if datasets.lower() == "retinal":
        msk_pred_np = np.squeeze(msk_pred, axis=0)
    else:
        msk_pred_np = _as_mask_numpy(msk_pred, thresh=threshold)
    plt.imsave(fname, msk_pred_np, cmap="gray")


def save_all_individually(
    img,
    msk,
    msk_pred,
    i: int,
    save_path: str,
    datasets: str,
    threshold: float = 0.5,
    test_data_name: str | None = None,
):
    """
    Save three separate files:
      * *_img.png   – RGB input
      * *_gt.png    – binarised ground‑truth
      * *_pred.png  – binarised prediction
    """
    _ensure_dir(save_path)
    prefix = f"{test_data_name}_" if test_data_name else ""
    base   = os.path.join(save_path, f"{prefix}{i}")

    img_np = _as_rgb_numpy(img)

    if datasets.lower() == "retinal":
        msk_np      = np.squeeze(msk,      axis=0)
        pred_np     = np.squeeze(msk_pred, axis=0)
    else:
        msk_np      = _as_mask_numpy(msk,      thresh=0.5)
        pred_np     = _as_mask_numpy(msk_pred, thresh=threshold)

    plt.imsave(f"{base}_img.png",  img_np)
    plt.imsave(f"{base}_gt.png",   msk_np,  cmap="gray")
    plt.imsave(f"{base}_pred.png", pred_np, cmap="gray")
