"""
MatAnyone2 inference wrapper for ComfyUI.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from comfy.utils import ProgressBar

from .constants import ckpt_path_matanyone2
from .src.matanyone2.inference.inference_core import InferenceCore
from .src.matanyone2.utils.get_default_model import get_matanyone2_model
from .src.matanyone2.utils.inference_utils import gen_dilate, gen_erosion

base_dir = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_MODES = (
    "first_valid_then_propagate",
    "valid_per_frame_then_propagate",
)


def get_matanyone2_model_cached():
    """Load MatAnyone2 from the checkpoint directory."""
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


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.numel() == 0:
        return mask
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def _refine_mask(mask: torch.Tensor, r_erode: int, r_dilate: int) -> torch.Tensor:
    """Apply optional morphology and return a 0..255 HW tensor."""
    mask_np = (mask * 255).round().cpu().numpy().astype(np.uint8)
    if r_dilate > 0:
        mask_np = gen_dilate(mask_np, r_dilate, r_dilate)
    if r_erode > 0:
        mask_np = gen_erosion(mask_np, r_erode, r_erode)
    return torch.from_numpy(mask_np).float()


def prepare_mask_guidance(
    masks: torch.Tensor,
    video_length: int,
    mask_frame: int,
    frame_height: int,
    frame_width: int,
    mask_mode: str = "first_valid_then_propagate",
    mask_valid_threshold: float = 0.0,
    r_erode: int = 0,
    r_dilate: int = 0,
) -> tuple[list[Optional[torch.Tensor]], int]:
    """Map a mask batch to video frames and find the first usable mask.

    A one-item batch keeps legacy ``mask_frame`` placement. Multi-mask batches
    are frame-aligned from frame 0. A mask is valid only when at least one pixel
    is greater than ``mask_valid_threshold``; fully black masks are therefore
    treated as missing guidance.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"Unknown mask_mode {mask_mode!r}; expected one of {MASK_MODES}")
    if video_length < 1:
        raise ValueError("src_video must contain at least one frame")
    if not 0 <= mask_frame < video_length:
        raise ValueError(
            f"mask_frame must be between 0 and {video_length - 1}, got {mask_frame}"
        )

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks in BHW format, got {tuple(masks.shape)}")
    if masks.shape[0] < 1:
        raise ValueError("At least one mask is required")
    if masks.shape[0] > video_length:
        raise ValueError(
            f"Received {masks.shape[0]} masks for only {video_length} video frames"
        )

    masks = _normalize_mask(masks)
    if masks.shape[-2:] != (frame_height, frame_width):
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(frame_height, frame_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    frame_indices = [mask_frame] if masks.shape[0] == 1 else list(range(masks.shape[0]))
    guidance: list[Optional[torch.Tensor]] = [None] * video_length
    valid_indices: list[int] = []

    for batch_index, frame_index in enumerate(frame_indices):
        mask = masks[batch_index]
        if mask.numel() and float(mask.max().item()) > mask_valid_threshold:
            guidance[frame_index] = _refine_mask(mask, r_erode, r_dilate)
            valid_indices.append(frame_index)

    if not valid_indices:
        raise ValueError(
            "No valid mask found: every supplied mask is fully black or below "
            f"mask_valid_threshold={mask_valid_threshold}"
        )

    anchor_frame = valid_indices[0]
    if mask_mode == "first_valid_then_propagate":
        anchor_mask = guidance[anchor_frame]
        guidance = [None] * video_length
        guidance[anchor_frame] = anchor_mask

    return guidance, anchor_frame


def _processor_step(
    processor: InferenceCore,
    image: torch.Tensor,
    guidance_mask: Optional[torch.Tensor],
):
    if guidance_mask is None:
        return processor.step(image)
    return processor.step(
        image,
        guidance_mask.to(device),
        objects=[1],
    )


def _store_output(phas, frame_index, processor, output_prob):
    mask_out = processor.output_prob_to_mask(output_prob)
    phas[frame_index] = mask_out.unsqueeze(0).unsqueeze(0).cpu()


def inference_matanyone2(
    vframes: torch.Tensor,
    masks: torch.Tensor,
    processor: InferenceCore,
    mask_frame: int = 0,
    n_warmup: int = 10,
    r_erode: int = 0,
    r_dilate: int = 0,
    mask_mode: str = "first_valid_then_propagate",
    mask_valid_threshold: float = 0.0,
):
    """Run MatAnyone2 with first-valid or per-frame mask guidance.

    ``first_valid_then_propagate`` (default) uses only the first non-black mask
    and propagates it through the complete video. ``valid_per_frame_then_propagate``
    injects every valid frame-aligned mask; whenever a frame's mask is black or
    absent, MatAnyone2 falls back to normal propagation from its current memory.
    """
    length, _, height, width = vframes.shape
    guidance, anchor_frame = prepare_mask_guidance(
        masks,
        video_length=length,
        mask_frame=mask_frame,
        frame_height=height,
        frame_width=width,
        mask_mode=mask_mode,
        mask_valid_threshold=mask_valid_threshold,
        r_erode=r_erode,
        r_dilate=r_dilate,
    )
    anchor_mask = guidance[anchor_frame]
    assert anchor_mask is not None

    backward_steps = anchor_frame
    forward_steps = length - anchor_frame - 1
    total_len = n_warmup + forward_steps
    if backward_steps:
        total_len += n_warmup + backward_steps
    pbar = ProgressBar(total_len)
    current_step = 0
    phas = [None] * length

    repeated_frames = (
        vframes[anchor_frame].unsqueeze(0).repeat(n_warmup, 1, 1, 1).to(device)
    )
    output_prob = warming_up2(
        anchor_mask.to(device),
        processor,
        n_warmup,
        repeated_frames,
        pbar,
        current_step,
        total_len,
    )
    current_step += n_warmup
    _store_output(phas, anchor_frame, processor, output_prob)

    for frame_index in range(anchor_frame + 1, length):
        output_prob = _processor_step(
            processor,
            vframes[frame_index].to(device),
            guidance[frame_index],
        )
        _store_output(phas, frame_index, processor, output_prob)
        current_step += 1
        pbar.update_absolute(current_step, total_len)

    if anchor_frame > 0:
        backward_processor = type(processor)(
            processor.network,
            cfg=processor.network.cfg,
        )
        repeated_frames_back = (
            vframes[anchor_frame]
            .unsqueeze(0)
            .repeat(n_warmup, 1, 1, 1)
            .to(device)
        )
        warming_up2(
            anchor_mask.to(device),
            backward_processor,
            n_warmup,
            repeated_frames_back,
            pbar,
            current_step,
            total_len,
        )
        current_step += n_warmup

        for frame_index in range(anchor_frame - 1, -1, -1):
            output_prob = _processor_step(
                backward_processor,
                vframes[frame_index].to(device),
                guidance[frame_index],
            )
            _store_output(phas, frame_index, backward_processor, output_prob)
            current_step += 1
            pbar.update_absolute(current_step, total_len)

    return phas
