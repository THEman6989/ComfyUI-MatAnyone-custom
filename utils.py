import torch


def get_mask(
    foreground_mask: torch.Tensor | None = None,
    foreground_MASK: torch.Tensor | None = None,
):
    """Legacy single-mask adapter used by the original MatAnyone node."""
    return get_mask_batch(foreground_mask, foreground_MASK).squeeze()


def get_mask_batch(
    foreground_mask: torch.Tensor | None = None,
    foreground_MASK: torch.Tensor | None = None,
):
    """Return masks as a stable ``(frames, height, width)`` batch.

    ComfyUI's MASK input is normally BHW while IMAGE is BHWC. Keeping the batch
    dimension is essential for MatAnyone2's per-frame guidance mode.
    """
    if foreground_mask is None and foreground_MASK is None:
        raise ValueError("Please provide one mask image")

    if foreground_MASK is not None:
        mask = foreground_MASK
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.ndim != 3:
            raise ValueError(
                f"Expected MASK in HW, BHW or B1HW format, got {tuple(mask.shape)}"
            )
    else:
        assert foreground_mask is not None
        if foreground_mask.ndim != 4:
            raise ValueError(
                f"Expected IMAGE in BHWC format, got {tuple(foreground_mask.shape)}"
            )
        mask = img_to_mask(foreground_mask.permute(0, 3, 1, 2))[:, 0]
    return mask.float()


def img_to_mask(tensor: torch.Tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device)
    weights = weights.view(1, 3, 1, 1)
    grayscale = torch.sum(tensor * weights, dim=1, keepdim=True)
    return grayscale


def get_screen(batch_size: int, height: int, width: int, r: int, g: int, b: int):
    # Normalize RGB values to the range 0.0-1.0
    r_normalized = float(r) / 255.0
    g_normalized = float(g) / 255.0
    b_normalized = float(b) / 255.0

    rgb_image = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
    rgb_image[:, :, :, 0] = r_normalized  # Red channel (index 0)
    rgb_image[:, :, :, 1] = g_normalized  # Green channel (index 1)
    rgb_image[:, :, :, 2] = b_normalized  # Blue channel (index 2)
    return rgb_image
