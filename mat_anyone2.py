"""
MatAnyone2 inference wrapper for ComfyUI.
"""

from pathlib import Path

import numpy as np
import torch

from comfy.utils import ProgressBar

from .constants import ckpt_path_matanyone2
from .src.matanyone2.inference.inference_core import InferenceCore
from .src.matanyone2.utils.get_default_model import get_matanyone2_model
from .src.matanyone2.utils.inference_utils import gen_dilate, gen_erosion

base_dir = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_matanyone2_model_cached():
    """Loads MatAnyone2 model from checkpoint (user must place matanyone2.pth in checkpoint/)."""
    ckpt_path = base_dir / ckpt_path_matanyone2
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "MatAnyone2 checkpoint not found at %s. "
            "Download matanyone2.pth and place it in the checkpoint folder." % ckpt_path
        )
    return get_matanyone2_model(str(ckpt_path), device)


def warming_up2(
    mask_t, processor, n_warmup, repeated_frames, pbar=None, pbar_start=0, total_len=0
):
    output_prob = None
    objects = [1]
    for ti in range(n_warmup):
        image = repeated_frames[ti]
        if ti == 0:
            output_prob = processor.step(image, mask_t, objects=objects)
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            output_prob = processor.step(image, first_frame_pred=True)
        if pbar is not None:
            pbar.update_absolute(pbar_start + ti, total_len)
    return output_prob


def inference_matanyone2(
    vframes: torch.Tensor,
    mask: torch.Tensor,
    processor: InferenceCore,
    mask_frame: int = 0,
    n_warmup: int = 10,
    r_erode: int = 0,
    r_dilate: int = 0,
):
    """
    Runs MatAnyone2 inference on video frames.

    Args:
        vframes: (T, C, H, W) RGB, float [0, 1]
        mask: (H, W) or (1, H, W) float [0, 1] or int
        processor: InferenceCore instance
        mask_frame: Frame index for mask
        n_warmup: Warmup iterations
        r_erode: Erosion kernel size
        r_dilate: Dilation kernel size

    Returns:
        List of alpha tensors (1, 1, H, W) per frame
    """
    mask = mask.squeeze().float()
    if mask.max() <= 1.0:
        mask = (mask * 255).clamp(0, 255)
    mask_np = mask.cpu().numpy().astype(np.uint8)

    if r_dilate > 0:
        mask_np = gen_dilate(mask_np, r_dilate, r_dilate)
    if r_erode > 0:
        mask_np = gen_erosion(mask_np, r_erode, r_erode)

    mask_t = torch.from_numpy(mask_np).float().to(device)
    length = vframes.shape[0]

    if mask_frame >= length:
        raise ValueError(f"Expected 0 <= x < {length}, got {mask_frame}")

    phas = [None] * length

    if mask_frame == 0:
        total_len = n_warmup + length
        pbar = ProgressBar(total_len)
        repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1).to(device)

        output_prob = warming_up2(
            mask_t, processor, n_warmup, repeated_frames, pbar, 0, total_len
        )
        mask_out = processor.output_prob_to_mask(output_prob)
        phas[0] = mask_out.unsqueeze(0).unsqueeze(0)

        for ti in range(1, length):
            image = vframes[ti].to(device)
            output_prob = processor.step(image)
            mask_out = processor.output_prob_to_mask(output_prob)
            phas[ti] = mask_out.unsqueeze(0).unsqueeze(0).cpu()
            pbar.update_absolute(n_warmup + ti, total_len)

    elif mask_frame == length - 1:
        total_len = n_warmup + length
        pbar = ProgressBar(total_len)
        reversed_vframes = torch.flip(vframes, dims=[0])
        repeated_frames = (
            reversed_vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1).to(device)
        )

        output_prob = warming_up2(
            mask_t, processor, n_warmup, repeated_frames, pbar, 0, total_len
        )
        mask_out = processor.output_prob_to_mask(output_prob)
        phas[mask_frame] = mask_out.unsqueeze(0).unsqueeze(0).cpu()

        for ti in range(1, length):
            image = reversed_vframes[ti].to(device)
            output_prob = processor.step(image)
            mask_out = processor.output_prob_to_mask(output_prob)
            phas[length - 1 - ti] = mask_out.unsqueeze(0).unsqueeze(0).cpu()
            pbar.update_absolute(n_warmup + ti, total_len)

    elif 0 < mask_frame < length - 1:
        # Two passes: forward then backward
        total_len = n_warmup + (length - mask_frame) + n_warmup + mask_frame
        pbar = ProgressBar(total_len)
        current_step = 0

        # Pass 1: Forward
        repeated_frames = (
            vframes[mask_frame].unsqueeze(0).repeat(n_warmup, 1, 1, 1).to(device)
        )
        output_prob = warming_up2(
            mask_t, processor, n_warmup, repeated_frames, pbar, current_step, total_len
        )
        current_step += n_warmup
        mask_out = processor.output_prob_to_mask(output_prob)
        phas[mask_frame] = mask_out.unsqueeze(0).unsqueeze(0).cpu()

        for ti in range(mask_frame + 1, length):
            image = vframes[ti].to(device)
            output_prob = processor.step(image)
            mask_out = processor.output_prob_to_mask(output_prob)
            phas[ti] = mask_out.unsqueeze(0).unsqueeze(0).cpu()
            pbar.update_absolute(current_step, total_len)
            current_step += 1

        # Pass 2: Backward
        processor = type(processor)(processor.network, cfg=processor.network.cfg)
        repeated_frames_back = (
            vframes[mask_frame].unsqueeze(0).repeat(n_warmup, 1, 1, 1).to(device)
        )

        output_prob = warming_up2(
            mask_t,
            processor,
            n_warmup,
            repeated_frames_back,
            pbar,
            current_step,
            total_len,
        )
        current_step += n_warmup

        reversed_vframes_backward = torch.flip(vframes[:mask_frame], dims=[0])

        for ti in range(mask_frame):
            image = reversed_vframes_backward[ti].to(device)
            output_prob = processor.step(image)
            mask_out = processor.output_prob_to_mask(output_prob)
            phas[mask_frame - 1 - ti] = mask_out.unsqueeze(0).unsqueeze(0).cpu()
            pbar.update_absolute(current_step, total_len)
            current_step += 1

    return phas
