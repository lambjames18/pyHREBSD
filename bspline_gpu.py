"""Weights and derivatives of spline orders 0 to 7."""

import os
import torch
from typing import List, Optional, Tuple
from enum import Enum
# from .jit_utils import square, cube, pow4, pow5, pow6, pow7, sub2ind_list, make_sign, cartesian_prod, pad_list_int, floor_div
Tensor = torch.Tensor

### utils.py contents ###

def fake_decorator(*a, **k):
    if len(a) == 1 and not k:
        return a[0]
    else:
        return fake_decorator


def make_list(x, n=None, **kwargs):
    """Ensure that the input  is a list (of a given size)

    Parameters
    ----------
    x : list or tuple or scalar
        Input object
    n : int, optional
        Required length
    default : scalar, optional
        Value to right-pad with. Use last value of the input by default.

    Returns
    -------
    x : list
    """
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n and len(x) < n:
        default = kwargs.get('default', x[-1])
        x = x + [default] * max(0, n - len(x))
    return x


def expanded_shape(*shapes, side='left'):
    """Expand input shapes according to broadcasting rules

    Parameters
    ----------
    *shapes : sequence[int]
        Input shapes
    side : {'left', 'right'}, default='left'
        Side to add singleton dimensions.

    Returns
    -------
    shape : tuple[int]
        Output shape

    Raises
    ------
    ValueError
        If shapes are not compatible for broadcast.

    """
    def error(s0, s1):
        raise ValueError('Incompatible shapes for broadcasting: {} and {}.'
                         .format(s0, s1))

    # 1. nb dimensions
    nb_dim = 0
    for shape in shapes:
        nb_dim = max(nb_dim, len(shape))

    # 2. enumerate
    shape = [1] * nb_dim
    for i, shape1 in enumerate(shapes):
        pad_size = nb_dim - len(shape1)
        ones = [1] * pad_size
        if side == 'left':
            shape1 = [*ones, *shape1]
        else:
            shape1 = [*shape1, *ones]
        shape = [max(s0, s1) if s0 == 1 or s1 == 1 or s0 == s1
                 else error(s0, s1) for s0, s1 in zip(shape, shape1)]

    return tuple(shape)


def matvec(mat, vec, out=None):
    """Matrix-vector product (supports broadcasting)

    Parameters
    ----------
    mat : (..., M, N) tensor
        Input matrix.
    vec : (..., N) tensor
        Input vector.
    out : (..., M) tensor, optional
        Placeholder for the output tensor.

    Returns
    -------
    mv : (..., M) tensor
        Matrix vector product of the inputs

    """
    vec = vec[..., None]
    if out is not None:
        out = out[..., None]

    mv = torch.matmul(mat, vec, out=out)
    mv = mv[..., 0]
    if out is not None:
        out = out[..., 0]

    return mv


def _compare_versions(version1, mode, version2):
    for v1, v2 in zip(version1, version2):
        if mode in ('gt', '>'):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('ge', '>='):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('lt', '<'):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        elif mode in ('le', '<='):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
    if mode in ('gt', 'lt', '>', '<'):
        return False
    else:
        return True


def torch_version(mode, version):
    """Check torch version

    Parameters
    ----------
    mode : {'<', '<=', '>', '>='}
    version : tuple[int]

    Returns
    -------
    True if "torch.version <mode> version"

    """
    current_version, *cuda_variant = torch.__version__.split('+')
    major, minor, patch, *_ = current_version.split('.')
    # strip alpha tags
    for x in 'abcdefghijklmnopqrstuvwxy':
        if x in patch:
            patch = patch[:patch.index(x)]
    current_version = (int(major), int(minor), int(patch))
    version = make_list(version)
    return _compare_versions(current_version, mode, version)



### jit_utils.py contents ###

def set_jit_enabled(enabled: bool):
    """ Enables/disables JIT """
    if torch.__version__ < "1.7":
        torch.jit._enabled = enabled
    else:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()


def jit_enabled():
    """ Returns whether JIT is enabled """
    if torch.__version__ < "1.7":
        return torch.jit._enabled
    else:
        return torch.jit._state._enabled.enabled


@torch.jit.script
def pad_list_int(x: List[int], dim: int) -> List[int]:
    if len(x) < dim:
        x = x + x[-1:] * (dim - len(x))
    if len(x) > dim:
        x = x[:dim]
    return x


@torch.jit.script
def pad_list_float(x: List[float], dim: int) -> List[float]:
    if len(x) < dim:
        x = x + x[-1:] * (dim - len(x))
    if len(x) > dim:
        x = x[:dim]
    return x


@torch.jit.script
def pad_list_str(x: List[str], dim: int) -> List[str]:
    if len(x) < dim:
        x = x + x[-1:] * (dim - len(x))
    if len(x) > dim:
        x = x[:dim]
    return x


@torch.jit.script
def list_any(x: List[bool]) -> bool:
    for elem in x:
        if elem:
            return True
    return False


@torch.jit.script
def list_all(x: List[bool]) -> bool:
    for elem in x:
        if not elem:
            return False
    return True


@torch.jit.script
def list_prod_int(x: List[int]) -> int:
    if len(x) == 0:
        return 1
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 * x1
    return x0


@torch.jit.script
def list_sum_int(x: List[int]) -> int:
    if len(x) == 0:
        return 1
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 + x1
    return x0


@torch.jit.script
def list_prod_tensor(x: List[Tensor]) -> Tensor:
    if len(x) == 0:
        empty: List[int] = []
        return torch.ones(empty)
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 * x1
    return x0


@torch.jit.script
def list_sum_tensor(x: List[Tensor]) -> Tensor:
    if len(x) == 0:
        empty: List[int] = []
        return torch.ones(empty)
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 + x1
    return x0


@torch.jit.script
def list_reverse_int(x: List[int]) -> List[int]:
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]


@torch.jit.script
def list_cumprod_int(x: List[int], reverse: bool = False,
                     exclusive: bool = False) -> List[int]:
    if len(x) == 0:
        lx: List[int] = []
        return lx
    if reverse:
        x = list_reverse_int(x)

    x0 = 1 if exclusive else x[0]
    lx = [x0]
    all_x = x[:-1] if exclusive else x[1:]
    for x1 in all_x:
        x0 = x0 * x1
        lx.append(x0)
    if reverse:
        lx = list_reverse_int(lx)
    return lx


@torch.jit.script
def movedim1(x, source: int, destination: int):
    dim = x.dim()
    source = dim + source if source < 0 else source
    destination = dim + destination if destination < 0 else destination
    permutation = [d for d in range(dim)]
    permutation = permutation[:source] + permutation[source+1:]
    permutation = permutation[:destination] + [source] + permutation[destination:]
    return x.permute(permutation)


@torch.jit.script
def sub2ind(subs, shape: List[int]):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D, ...) tensor
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) list[int]
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    subs = subs.unbind(0)
    ind = subs[-1]
    subs = subs[:-1]
    ind = ind.clone()
    stride = list_cumprod_int(shape[1:], reverse=True, exclusive=False)
    for i, s in zip(subs, stride):
        ind += i * s
    return ind


@torch.jit.script
def sub2ind_list(subs: List[Tensor], shape: List[int]):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D,) list[tensor]
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) list[int]
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    ind = subs[-1]
    subs = subs[:-1]
    ind = ind.clone()
    stride = list_cumprod_int(shape[1:], reverse=True, exclusive=False)
    for i, s in zip(subs, stride):
        ind += i * s
    return ind

# floor_divide returns wrong results for negative values, because it truncates
# instead of performing a proper floor. In recent version of pytorch, it is
# advised to use div(..., rounding_mode='trunc'|'floor') instead.
# Here, we only use floor_divide on positive values so we do not care.
if torch_version('>=', [1, 8]):
    @torch.jit.script
    def floor_div(x, y) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='floor')
    @torch.jit.script
    def floor_div_int(x, y: int) -> torch.Tensor:
        return torch.div(x, y, rounding_mode='floor')
else:
    @torch.jit.script
    def floor_div(x, y) -> torch.Tensor:
        return (x / y).floor_()
    @torch.jit.script
    def floor_div_int(x, y: int) -> torch.Tensor:
        return (x / y).floor_()


@torch.jit.script
def ind2sub(ind, shape: List[int]):
    """Convert linear indices into sub indices (i, j, k).

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    ind : tensor_like
        Linear indices
    shape : (D,) vector_like
        Size of each dimension.

    Returns
    -------
    subs : (D, ...) tensor
        Sub-indices.
    """
    stride = list_cumprod_int(shape, reverse=True, exclusive=True)
    sub = ind.new_empty([len(shape)] + ind.shape)
    sub.copy_(ind)
    for d in range(len(shape)):
        if d > 0:
            sub[d] = torch.remainder(sub[d], stride[d-1])
        sub[d] = floor_div_int(sub[d], stride[d])
    return sub


@torch.jit.script
def inbounds_mask_3d(extrapolate: int, gx, gy, gz, nx: int, ny: int, nz: int) \
        -> Optional[Tensor]:
    # mask of inbounds voxels
    mask: Optional[Tensor] = None
    if extrapolate in (0, 2):  # no / hist
        tiny = 5e-2
        threshold = tiny
        if extrapolate == 2:
            threshold = 0.5 + tiny
        mask = ((gx > -threshold) & (gx < nx - 1 + threshold) &
                (gy > -threshold) & (gy < ny - 1 + threshold) &
                (gz > -threshold) & (gz < nz - 1 + threshold))
        return mask
    return mask


@torch.jit.script
def inbounds_mask_2d(extrapolate: int, gx, gy, nx: int, ny: int) \
        -> Optional[Tensor]:
    # mask of inbounds voxels
    mask: Optional[Tensor] = None
    if extrapolate in (0, 2):  # no / hist
        tiny = 5e-2
        threshold = tiny
        if extrapolate == 2:
            threshold = 0.5 + tiny
        mask = ((gx > -threshold) & (gx < nx - 1 + threshold) &
                (gy > -threshold) & (gy < ny - 1 + threshold))
        return mask
    return mask


@torch.jit.script
def inbounds_mask_1d(extrapolate: int, gx, nx: int) -> Optional[Tensor]:
    # mask of inbounds voxels
    mask: Optional[Tensor] = None
    if extrapolate in (0, 2):  # no / hist
        tiny = 5e-2
        threshold = tiny
        if extrapolate == 2:
            threshold = 0.5 + tiny
        mask = (gx > -threshold) & (gx < nx - 1 + threshold)
        return mask
    return mask


@torch.jit.script
def make_sign(sign: List[Optional[Tensor]]) -> Optional[Tensor]:
    is_none : List[bool] = [s is None for s in sign]
    if list_all(is_none):
        return None
    filt_sign: List[Tensor] = []
    for s in sign:
        if s is not None:
            filt_sign.append(s)
    return list_prod_tensor(filt_sign)


@torch.jit.script
def square(x):
    return x * x


@torch.jit.script
def square_(x):
    return x.mul_(x)


@torch.jit.script
def cube(x):
    return x * x * x


@torch.jit.script
def cube_(x):
    return square_(x).mul_(x)


@torch.jit.script
def pow4(x):
    return square(square(x))


@torch.jit.script
def pow4_(x):
    return square_(square_(x))


@torch.jit.script
def pow5(x):
    return x * pow4(x)


@torch.jit.script
def pow5_(x):
    return pow4_(x).mul_(x)


@torch.jit.script
def pow6(x):
    return square(cube(x))


@torch.jit.script
def pow6_(x):
    return square_(cube_(x))


@torch.jit.script
def pow7(x):
    return pow6(x) * x


@torch.jit.script
def pow7_(x):
    return pow6_(x).mul_(x)


@torch.jit.script
def dot(x, y, dim: int = -1, keepdim: bool = False):
    """(Batched) dot product along a dimension"""
    x = movedim1(x, dim, -1).unsqueeze(-2)
    y = movedim1(y, dim, -1).unsqueeze(-1)
    d = torch.matmul(x, y).squeeze(-1).squeeze(-1)
    if keepdim:
        d.unsqueeze(dim)
    return d


@torch.jit.script
def dot_multi(x, y, dim: List[int], keepdim: bool = False):
    """(Batched) dot product along a dimension"""
    for d in dim:
        x = movedim1(x, d, -1)
        y = movedim1(y, d, -1)
    x = x.reshape(x.shape[:-len(dim)] + [1, -1])
    y = y.reshape(x.shape[:-len(dim)] + [-1, 1])
    dt = torch.matmul(x, y).squeeze(-1).squeeze(-1)
    if keepdim:
        for d in dim:
            dt.unsqueeze(d)
    return dt



# cartesian_prod takes multiple inout tensors as input in eager mode
# but takes a list of tensor in jit mode. This is a helper that works
# in both cases.
if not int(os.environ.get('PYTORCH_JIT', '1')):
    cartesian_prod = lambda x: torch.cartesian_prod(*x)
    if torch_version('>=', (1, 10)):
        def meshgrid_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x, indexing='ij')
        def meshgrid_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x, indexing='xy')
    else:
        def meshgrid_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x)
        def meshgrid_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(*x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid

else:
    cartesian_prod = torch.cartesian_prod
    if torch_version('>=', (1, 10)):
        @torch.jit.script
        def meshgrid_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='ij')
        @torch.jit.script
        def meshgrid_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='xy')
    else:
        @torch.jit.script
        def meshgrid_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x)
        @torch.jit.script
        def meshgrid_xyt(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid


meshgrid = meshgrid_ij


# In torch < 1.6, div applied to integer tensor performed a floor_divide
# In torch > 1.6, it performs a true divide.
# Floor division must be done using `floor_divide`, but it was buggy
# until torch 1.13 (it was doing a trunc divide instead of a floor divide).
# There was at some point a deprecation warning for floor_divide, but it
# seems to have been lifted afterwards. In torch >= 1.13, floor_divide
# performs a correct floor division.
# Since we only apply floor_divide ot positive values, we are fine.
if torch_version('<', (1, 6)):
    floor_div = torch.div
else:
    floor_div = torch.floor_divide


### bounds.py contents ###

class BoundType(Enum):
    zero = zeros = 0
    replicate = nearest = 1
    dct1 = mirror = 2
    dct2 = reflect = 3
    dst1 = antimirror = 4
    dst2 = antireflect = 5
    dft = wrap = 6


class ExtrapolateType(Enum):
    no = 0     # threshold: (0, n-1)
    yes = 1
    hist = 2   # threshold: (-0.5, n-0.5)


@torch.jit.script
class Bound:

    def __init__(self, bound_type: int = 3):
        self.type = bound_type

    def index(self, i, n: int):
        if self.type in (0, 1):  # zero / replicate
            return i.clamp(min=0, max=n-1)
        elif self.type in (3, 5):  # dct2 / dst2
            n2 = n * 2
            i = torch.where(i < 0, (-i-1).remainder(n2).neg().add(n2 - 1),
                            i.remainder(n2))
            i = torch.where(i >= n, -i + (n2 - 1), i)
            return i
        elif self.type == 2:  # dct1
            if n == 1:
                return torch.zeros(i.shape, dtype=i.dtype, device=i.device)
            else:
                n2 = (n - 1) * 2
                i = i.abs().remainder(n2)
                i = torch.where(i >= n, -i + n2, i)
                return i
        elif self.type == 4:  # dst1
            n2 = 2 * (n + 1)
            first = torch.zeros([1], dtype=i.dtype, device=i.device)
            last = torch.full([1], n - 1, dtype=i.dtype, device=i.device)
            i = torch.where(i < 0, -i - 2, i)
            i = i.remainder(n2)
            i = torch.where(i > n, -i + (n2 - 2), i)
            i = torch.where(i == -1, first, i)
            i = torch.where(i == n, last, i)
            return i
        elif self.type == 6:  # dft
            return i.remainder(n)
        else:
            return i

    def transform(self, i, n: int) -> Optional[Tensor]:
        if self.type == 4:  # dst1
            if n == 1:
                return None
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            n2 = 2 * (n + 1)
            i = torch.where(i < 0, -i + (n-1), i)
            i = i.remainder(n2)
            x = torch.where(i == 0, zero, one)
            x = torch.where(i.remainder(n + 1) == n, zero, x)
            i = floor_div(i, n+1)
            x = torch.where(torch.remainder(i, 2) > 0, -x, x)
            return x
        elif self.type == 5:  # dst2
            i = torch.where(i < 0, n - 1 - i, i)
            x = torch.ones([1], dtype=torch.int8, device=i.device)
            i = floor_div(i, n)
            x = torch.where(torch.remainder(i, 2) > 0, -x, x)
            return x
        elif self.type == 0:  # zero
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            outbounds = ((i < 0) | (i >= n))
            x = torch.where(outbounds, zero, one)
            return x
        else:
            return None


### splines.py contents ###

class InterpolationType(Enum):
    nearest = zeroth = 0
    linear = first = 1
    quadratic = second = 2
    cubic = third = 3
    fourth = 4
    fifth = 5
    sixth = 6
    seventh = 7


@torch.jit.script
class Spline:

    def __init__(self, order: int = 1):
        self.order = order

    def weight(self, x):
        w = self.fastweight(x)
        zero = torch.zeros([1], dtype=x.dtype, device=x.device)
        w = torch.where(x.abs() >= (self.order + 1)/2, zero, w)
        return w

    def fastweight(self, x):
        if self.order == 0:
            return torch.ones(x.shape, dtype=x.dtype, device=x.device)
        x = x.abs()
        if self.order == 1:
            return 1 - x
        if self.order == 2:
            x_low = 0.75 - square(x)
            x_up = 0.5 * square(1.5 - x)
            return torch.where(x < 0.5, x_low, x_up)
        if self.order == 3:
            x_low = (x * x * (x - 2.) * 3. + 4.) / 6.
            x_up = cube(2. - x) / 6.
            return torch.where(x < 1., x_low, x_up)
        if self.order == 4:
            x_low = square(x)
            x_low = x_low * (x_low * 0.25 - 0.625) + 115. / 192.
            x_mid = x * (x * (x * (5. - x) / 6. - 1.25) + 5./24.) + 55./96.
            x_up = pow4(x - 2.5) / 24.
            return torch.where(x < 0.5, x_low, torch.where(x < 1.5, x_mid, x_up))
        if self.order == 5:
            x_low = square(x)
            x_low = x_low * (x_low * (0.25 - x / 12.) - 0.5) + 0.55
            x_mid = x * (x * (x * (x * (x / 24. - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
            x_up = pow5(3 - x) / 120.
            return torch.where(x < 1., x_low, torch.where(x < 2., x_mid, x_up))
        if self.order == 6:
            x_low = square(x)
            x_low = x_low * (x_low * (7./48. - x_low/36.) - 77./192.) + 5887./11520.
            x_mid_low = (x * (x * (x * (x * (x * (x / 48. - 7./48.) + 0.328125)
                         - 35./288.) - 91./256.) - 7./768.) + 7861./15360.)
            x_mid_up = (x * (x * (x * (x * (x * (7./60. - x / 120.) - 0.65625)
                        + 133./72.) - 2.5703125) + 1267./960.) + 1379./7680.)
            x_up = pow6(x - 3.5) / 720.
            return torch.where(x < .5, x_low,
                               torch.where(x < 1.5, x_mid_low,
                                           torch.where(x < 2.5, x_mid_up, x_up)))
        if self.order == 7:
            x_low = square(x)
            x_low = (x_low * (x_low * (x_low * (x / 144. - 1./36.)
                     + 1./9.) - 1./3.) + 151./315.)
            x_mid_low = (x * (x * (x * (x * (x * (x * (0.05 - x/240.) - 7./30.)
                         + 0.5) - 7./18.) - 0.1) - 7./90.) + 103./210.)
            x_mid_up = (x * (x * (x * (x * (x * (x * (x / 720. - 1./36.)
                        + 7./30.) - 19./18.) + 49./18.) - 23./6.) + 217./90.)
                        - 139./630.)
            x_up = pow7(4 - x) / 5040.
            return torch.where(x < 1., x_low,
                               torch.where(x < 2., x_mid_low,
                                           torch.where(x < 3., x_mid_up, x_up)))
        raise NotImplementedError

    def grad(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        g = self.fastgrad(x)
        zero = torch.zeros([1], dtype=x.dtype, device=x.device)
        g = torch.where(x.abs() >= (self.order + 1)/2, zero, g)
        return g

    def fastgrad(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        return self._fastgrad(x.abs()).mul(x.sign())

    def _fastgrad(self, x):
        if self.order == 1:
            return torch.ones(x.shape, dtype=x.dtype, device=x.device)
        if self.order == 2:
            return torch.where(x < 0.5, -2*x, x - 1.5)
        if self.order == 3:
            g_low = x * (x * 1.5 - 2)
            g_up = -0.5 * square(2 - x)
            return torch.where(x < 1, g_low, g_up)
        if self.order == 4:
            g_low = x * (square(x) - 1.25)
            g_mid = x * (x * (x * (-2./3.) + 2.5) - 2.5) + 5./24.
            g_up = cube(2. * x - 5.) / 48.
            return torch.where(x < 0.5, g_low,
                               torch.where(x < 1.5, g_mid, g_up))
        if self.order == 5:
            g_low = x * (x * (x * (x * (-5./12.) + 1.)) - 1.)
            g_mid = x * (x * (x * (x * (5./24.) - 1.5) + 3.75) - 3.5) + 0.625
            g_up = pow4(x - 3.) / (-24.)
            return torch.where(x < 1, g_low,
                               torch.where(x < 2, g_mid, g_up))
        if self.order == 6:
            g_low = square(x)
            g_low = x * (g_low * (7./12.) - square(g_low) / 6. - 77./96.)
            g_mid_low = (x * (x * (x * (x * (x * 0.125 - 35./48.) + 1.3125)
                         - 35./96.) - 0.7109375) - 7./768.)
            g_mid_up = (x * (x * (x * (x * (x / (-20.) + 7./12.) - 2.625)
                        + 133./24.) - 5.140625) + 1267./960.)
            g_up = pow5(2*x - 7) / 3840.
            return torch.where(x < 0.5, g_low,
                               torch.where(x < 1.5, g_mid_low,
                                           torch.where(x < 2.5, g_mid_up,
                                                       g_up)))
        if self.order == 7:
            g_low = square(x)
            g_low = x * (g_low * (g_low * (x * (7./144.) - 1./6.) + 4./9.) - 2./3.)
            g_mid_low = (x * (x * (x * (x * (x * (x * (-7./240.) + 3./10.)
                         - 7./6.) + 2.) - 7./6.) - 1./5.) - 7./90.)
            g_mid_up = (x * (x * (x * (x * (x * (x * (7./720.) - 1./6.)
                        + 7./6.) - 38./9.) + 49./6.) - 23./3.) + 217./90.)
            g_up = pow6(x - 4) / (-720.)
            return torch.where(x < 1, g_low,
                               torch.where(x < 2, g_mid_low,
                                           torch.where(x < 3, g_mid_up, g_up)))
        raise NotImplementedError

    def hess(self, x):
        if self.order == 0:
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        h = self.fasthess(x)
        zero = torch.zeros([1], dtype=x.dtype, device=x.device)
        h = torch.where(x.abs() >= (self.order + 1)/2, zero, h)
        return h

    def fasthess(self, x):
        if self.order in (0, 1):
            return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        x = x.abs()
        if self.order == 2:
            one = torch.ones([1], dtype=x.dtype, device=x.device)
            return torch.where(x < 0.5, -2 * one, one)
        if self.order == 3:
            return torch.where(x < 1, 3. * x - 2., 2. - x)
        if self.order == 4:
            return torch.where(x < 0.5, 3. * square(x) - 1.25,
                               torch.where(x < 1.5, x * (-2. * x + 5.) - 2.5,
                                           square(2. * x - 5.) / 8.))
        if self.order == 5:
            h_low = square(x)
            h_low = - h_low * (x * (5./3.) - 3.) - 1.
            h_mid = x * (x * (x * (5./6.) - 9./2.) + 15./2.) - 7./2.
            h_up = 9./2. - x * (x * (x/6. - 3./2.) + 9./2.)
            return torch.where(x < 1, h_low,
                               torch.where(x < 2, h_mid, h_up))
        if self.order == 6:
            h_low = square(x)
            h_low = - h_low * (h_low * (5./6) - 7./4.) - 77./96.
            h_mid_low = (x * (x * (x * (x * (5./8.) - 35./12.) + 63./16.)
                         - 35./48.) - 91./128.)
            h_mid_up = -(x * (x * (x * (x/4. - 7./3.) + 63./8.) - 133./12.)
                         + 329./64.)
            h_up = (x * (x * (x * (x/24. - 7./12.) + 49./16.) - 343./48.)
                    + 2401./384.)
            return torch.where(x < 0.5, h_low,
                               torch.where(x < 1.5, h_mid_low,
                                           torch.where(x < 2.5, h_mid_up,
                                                       h_up)))
        if self.order == 7:
            h_low = square(x)
            h_low = h_low * (h_low*(x * (7./24.) - 5./6.) + 4./3.) - 2./3.
            h_mid_low = - (x * (x * (x * (x * (x * (7./40.) - 3./2.) + 14./3.)
                           - 6.) + 7./3.) + 1./5.)
            h_mid_up = (x * (x * (x * (x * (x * (7./120.) - 5./6.) + 14./3.)
                        - 38./3.) + 49./3.) - 23./3.)
            h_up = - (x * (x * (x * (x * (x/120. - 1./6.) + 4./3.) - 16./3.)
                      + 32./3.) - 128./15.)
            return torch.where(x < 1, h_low,
                               torch.where(x < 2, h_mid_low,
                                           torch.where(x < 3, h_mid_up,
                                                       h_up)))
        raise NotImplementedError


### nd.py contents ###

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


@torch.jit.script
def grad(inp, grid, bound: List[Bound], spline: List[Spline],
         extrapolate: int = 1):
    """
    inp: (B, C, *ishape) tensor
    grid: (B, *oshape, D) tensor
    bound: List{D}[Bound] tensor
    spline: List{D}[Spline] tensor
    extrapolate: int
    returns: (B, C, *oshape, D) tensor
    """

    # Get shape of input and grid, batch size, and number of channels
    dim = grid.shape[-1]
    shape = list(inp.shape[-dim:])
    oshape = list(grid.shape[-dim-1:-1])
    batch = max(inp.shape[0], grid.shape[0])
    channel = inp.shape[1]

    # Reshape input and grid to collapse spatial dimensions
    grid = grid.reshape([grid.shape[0], -1, grid.shape[-1]])
    inp = inp.reshape([inp.shape[0], inp.shape[1], -1])
    mask = inbounds_mask(extrapolate, grid, shape)

    # precompute weights along each dimension
    weights, grads, _, coords, signs = get_weights(grid, bound, spline, shape,
                                                   grad=True)

    # initialize
    out = torch.zeros([batch, channel, grid.shape[1], dim],
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
        out0 = inp.gather(-1, idx)

        # apply sign
        sign0: List[Optional[Tensor]] = [sgn[n] for sgn, n in zip(signs, nodes)]
        sign1: Optional[Tensor] = make_sign(sign0)
        if sign1 is not None:
            out0 = out0 * sign1.unsqueeze(1)

        for d in range(dim):
            out1 = out0.clone()
            # apply weights
            for dd, (weight, grad1, n) in enumerate(zip(weights, grads, nodes)):
                if d == dd:
                    grad11 = grad1[n]
                    if grad11 is not None:
                        out1 = out1 * grad11.unsqueeze(1)
                else:
                    out1 = out1 * weight[n].unsqueeze(1)

            # accumulate
            out.unbind(-1)[d].add_(out1)

    # out-of-bounds mask
    if mask is not None:
        out = out * mask.unsqueeze(-1)

    out = out.reshape(list(out.shape[:2]) + oshape + list(out.shape[-1:]))
    return out


### pull.py contents ###

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
    return pull(inp, grid, bound_fn, spline_fn, extrapolate)