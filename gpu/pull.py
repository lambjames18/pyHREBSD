
"""
Non-differentiable forward/backward components.
These components are put together in `interpol.autograd` to generate
differentiable functions.
"""

import torch
from typing import List
from .jit_utils import pad_list_int
from .bounds import Bound
from .splines import Spline
from . import nd
Tensor = torch.Tensor


@torch.jit.script
def make_bound(bound: List[int]) -> List[Bound]:
    return [Bound(b) for b in bound]


@torch.jit.script
def make_spline(spline: List[int]) -> List[Spline]:
    return [Spline(s) for s in spline]


@torch.jit.script
def grid_pull(inp, grid, bound: List[int], interpolation: List[int],
              extrapolate: int):
    """
    inp: (B, C, *spatial_in) tensor
    grid: (B, *spatial_out, D) tensor
    bound: List{D}[int] tensor
    interpolation: List{D}[int]
    extrapolate: int
    returns: (B, C, *spatial_out) tensor
    """
    dim = grid.shape[-1]
    bound = pad_list_int(bound, dim)
    interpolation = pad_list_int(interpolation, dim)
    bound_fn = make_bound(bound)
    spline_fn = make_spline(interpolation)
    return nd.pull(inp, grid, bound_fn, spline_fn, extrapolate)
