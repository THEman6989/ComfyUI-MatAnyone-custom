"""
Auxiliary output modules for MatAnyone2.
"""

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..utils.tensor_utils import aggregate
from .group_modules import GConv2d


class LinearPredictor(nn.Module):
    """Linear predictor for auxiliary loss."""

    def __init__(self, x_dim: int, pix_dim: int):
        super().__init__()
        self.projection = GConv2d(x_dim, pix_dim + 1, kernel_size=1)

    def forward(self, pix_feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        num_objects = x.shape[1]
        x = self.projection(x)
        pix_feat = pix_feat.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        logits = (pix_feat * x[:, :, :-1]).sum(dim=2) + x[:, :, -1]
        return logits


class DirectPredictor(nn.Module):
    """Direct predictor for auxiliary loss."""

    def __init__(self, x_dim: int):
        super().__init__()
        self.projection = GConv2d(x_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).squeeze(2)


class AuxComputer(nn.Module):
    """Computes auxiliary outputs for auxiliary losses."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        use_sensory_aux = cfg.model.aux_loss.sensory.enabled
        self.use_query_aux = cfg.model.aux_loss.query.enabled
        self.use_sensory_aux = use_sensory_aux
        sensory_dim = cfg.model.sensory_dim
        embed_dim = cfg.model.embed_dim

        if use_sensory_aux:
            self.sensory_aux = LinearPredictor(sensory_dim, embed_dim)

    def _aggregate_with_selector(
        self, logits: torch.Tensor, selector: torch.Tensor
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
        return aggregate(prob, dim=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        aux_input: Dict[str, torch.Tensor],
        selector: torch.Tensor,
        seg_pass=False,
    ) -> Dict[str, torch.Tensor]:
        sensory = aux_input["sensory"]
        q_logits = aux_input["q_logits"]
        aux_output = {}
        aux_output["attn_mask"] = aux_input["attn_mask"]

        if self.use_sensory_aux:
            logits = self.sensory_aux(pix_feat, sensory)
            aux_output["sensory_logits"] = self._aggregate_with_selector(
                logits, selector
            )
        if self.use_query_aux:
            aux_output["q_logits"] = self._aggregate_with_selector(
                torch.stack(q_logits, dim=2),
                selector.unsqueeze(2) if selector is not None else None,
            )
        return aux_output

    def compute_mask(
        self,
        aux_input: Dict[str, torch.Tensor],
        selector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        q_logits = aux_input["q_logits"]
        aux_output = {}
        aux_output["q_logits"] = self._aggregate_with_selector(
            torch.stack(q_logits, dim=2),
            selector.unsqueeze(2) if selector is not None else None,
        )
        return aux_output
