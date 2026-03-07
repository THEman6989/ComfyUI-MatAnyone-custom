"""
Loads MatAnyone2 model from checkpoint.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from ..model.matanyone2 import MatAnyone2


def get_matanyone2_model(ckpt_path: str, device=None) -> MatAnyone2:
    """Loads MatAnyone2 model from checkpoint path."""
    config_dir = Path(__file__).resolve().parent.parent / "config"

    # Load base model config and eval config
    model_cfg = OmegaConf.load(config_dir / "model" / "base.yaml")
    eval_cfg = OmegaConf.load(config_dir / "eval_matanyone_config.yaml")

    # Clean up hydra/defaults from eval_cfg if they exist
    if "defaults" in eval_cfg:
        del eval_cfg["defaults"]
    if "hydra" in eval_cfg:
        del eval_cfg["hydra"]

    # Merge them manually as hydra did
    cfg = OmegaConf.merge(eval_cfg, {"model": model_cfg})
    OmegaConf.resolve(cfg)

    with open_dict(cfg):
        cfg["weights"] = ckpt_path

    if device is not None:
        matanyone2 = MatAnyone2(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(cfg.weights, map_location=device)
    else:
        matanyone2 = MatAnyone2(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(cfg.weights)

    matanyone2.load_weights(model_weights)

    return matanyone2
