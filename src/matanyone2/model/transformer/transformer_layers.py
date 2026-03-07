"""
Transformer layers for MatAnyone2.
"""

from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..channel_attn import CAResBlock


class SelfAttention(nn.Module):
    """Self-attention with optional positional encoding."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        add_pe_to_qkv: List[bool] = None,
    ):
        super().__init__()
        if add_pe_to_qkv is None:
            add_pe_to_qkv = [True, True, False]
        self.self_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=batch_first
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(
        self,
        x: torch.Tensor,
        pe: torch.Tensor,
        attn_mask: bool = None,
        key_padding_mask: bool = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x

        r = x
        x = self.self_attn(
            q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]
        return r + self.dropout(x)


class CrossAttention(nn.Module):
    """Cross-attention with optional positional encoding."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        add_pe_to_qkv: List[bool] = None,
        residual: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        if add_pe_to_qkv is None:
            add_pe_to_qkv = [True, True, False]
        self.cross_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=batch_first
        )
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        x_pe: torch.Tensor,
        mem_pe: torch.Tensor,
        attn_mask: bool = None,
        *,
        need_weights: bool = False,
    ) -> tuple:
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem

        r = x
        x, weights = self.cross_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        if self.residual:
            return r + self.dropout(x), weights
        return self.dropout(x), weights


class FFN(nn.Module):
    """Feed-forward network."""

    def __init__(self, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)
        self.activation = (
            _get_activation_fn(activation)
            if isinstance(activation, str)
            else activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        return r + x


class PixelFFN(nn.Module):
    """Pixel-level feed-forward using CAResBlock."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: torch.Tensor, pixel_flat: torch.Tensor) -> torch.Tensor:
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs * num_objects, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2).contiguous()
        x = self.conv(pixel_flat)
        x = x.view(bs, num_objects, self.dim, h, w)
        return x


class OutputFFN(nn.Module):
    """Output projection FFN."""

    def __init__(self, dim_in: int, dim_out: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)
        self.activation = (
            _get_activation_fn(activation)
            if isinstance(activation, str)
            else activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not %s" % activation)
