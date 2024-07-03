"""Generic N-dimensional version: any combination of spline orders"""
import torch
from typing import List, Optional, Tuple
from .bounds import Bound
from .splines import Spline
from .jit_utils import sub2ind_list, make_sign, cartesian_prod
Tensor = torch.Tensor


@torch.jit.script
def inbounds_mask(extrapolate: int, grid, shape: List[int])\
        -> Optional[Tensor]:
    # mask of inbounds voxels
    mask: Optional[Tensor] = None
    if extrapolate in (0, 2):  # no / hist
        grid = grid.unsqueeze(1)
        tiny = 5e-2
        threshold = tiny
        if extrapolate == 2:
            threshold = 0.5 + tiny
        mask = torch.ones(grid.shape[:-1],
                          dtype=torch.bool, device=grid.device)
        for grid1, shape1 in zip(grid.unbind(-1), shape):
            mask = mask & (grid1 > -threshold)
            mask = mask & (grid1 < shape1 - 1 + threshold)
        return mask
    return mask


@torch.jit.script
def get_weights(grid, bound: List[Bound], spline: List[Spline],
                shape: List[int], grad: bool = False, hess: bool = False) \
        -> Tuple[List[List[Tensor]],
                 List[List[Optional[Tensor]]],
                 List[List[Optional[Tensor]]],
                 List[List[Tensor]],
                 List[List[Optional[Tensor]]]]:

    weights: List[List[Tensor]] = []
    grads: List[List[Optional[Tensor]]] = []
    hesss: List[List[Optional[Tensor]]] = []
    coords: List[List[Tensor]] = []
    signs: List[List[Optional[Tensor]]] = []
    for g, b, s, n in zip(grid.unbind(-1), bound, spline, shape):
        grid0 = (g - (s.order-1)/2).floor()
        dist0 = g - grid0
        grid0 = grid0.long()
        nb_nodes = s.order + 1
        subweights: List[Tensor] = []
        subcoords: List[Tensor] = []
        subgrads: List[Optional[Tensor]] = []
        subhesss: List[Optional[Tensor]] = []
        subsigns: List[Optional[Tensor]] = []
        for node in range(nb_nodes):
            grid1 = grid0 + node
            sign1: Optional[Tensor] = b.transform(grid1, n)
            subsigns.append(sign1)
            grid1 = b.index(grid1, n)
            subcoords.append(grid1)
            dist1 = dist0 - node
            weight1 = s.fastweight(dist1)
            subweights.append(weight1)
            grad1: Optional[Tensor] = None
            if grad:
                grad1 = s.fastgrad(dist1)
            subgrads.append(grad1)
            hess1: Optional[Tensor] = None
            if hess:
                hess1 = s.fasthess(dist1)
            subhesss.append(hess1)
        weights.append(subweights)
        coords.append(subcoords)
        signs.append(subsigns)
        grads.append(subgrads)
        hesss.append(subhesss)

    return weights, grads, hesss, coords, signs


@torch.jit.script
def pull(inp, grid, bound: List[Bound], spline: List[Spline],
         extrapolate: int = 1):
    """
    inp: (B, C, *ishape) tensor
    g: (B, *oshape, D) tensor
    bound: List{D}[Bound] tensor
    spline: List{D}[Spline] tensor
    extrapolate: int
    returns: (B, C, *oshape) tensor
    """

    dim = grid.shape[-1]
    shape = list(inp.shape[-dim:])
    oshape = list(grid.shape[-dim-1:-1])
    batch = max(inp.shape[0], grid.shape[0])
    channel = inp.shape[1]

    grid = grid.reshape([grid.shape[0], -1, grid.shape[-1]])
    inp = inp.reshape([inp.shape[0], inp.shape[1], -1])
    mask = inbounds_mask(extrapolate, grid, shape)

    # precompute weights along each dimension
    weights, _, _, coords, signs = get_weights(grid, bound, spline, shape, False, False)

    # initialize
    out = torch.zeros([batch, channel, grid.shape[1]],
                      dtype=inp.dtype, device=inp.device)

    # iterate across nodes/corners
    range_nodes = [torch.as_tensor([d for d in range(n)])
                   for n in [s.order + 1 for s in spline]]
    if dim == 1:
        # cartesian_prod does not work as expected when only one
        # element is provided
        all_nodes = range_nodes[0].unsqueeze(-1)
    else:
        all_nodes = cartesian_prod(range_nodes)
    for nodes in all_nodes:
        # gather
        idx = [c[n] for c, n in zip(coords, nodes)]
        idx = sub2ind_list(idx, shape).unsqueeze(1)
        idx = idx.expand([batch, channel, idx.shape[-1]])
        out1 = inp.gather(-1, idx)

        # apply sign
        sign0: List[Optional[Tensor]] = [sgn[n] for sgn, n in zip(signs, nodes)]
        sign1: Optional[Tensor] = make_sign(sign0)
        if sign1 is not None:
            out1 = out1 * sign1.unsqueeze(1)

        # apply weights
        for weight, n in zip(weights, nodes):
            out1 = out1 * weight[n].unsqueeze(1)

        # accumulate
        out = out + out1

    # out-of-bounds mask
    if mask is not None:
        out = out * mask

    out = out.reshape(list(out.shape[:2]) + oshape)
    return out
