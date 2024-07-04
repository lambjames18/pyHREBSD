import numpy as np
from scipy import interpolate
import torch


class Spline:
    def __init__(self, image, kx, ky, PC=None, subset_slice=None):
        # Inputs
        self.image = image
        self.kx = kx
        self.ky = ky
        if PC is None:
            self.PC = np.array(image.shape)[::-1] / 2
        else:
            self.PC = PC
        if subset_slice is None:
            self.subset_slice = (slice(None), slice(None))
        else:
            self.subset_slice = subset_slice

        # Create the spline
        x = np.arange(image.shape[1]) - PC[0]
        y = np.arange(image.shape[0]) - PC[1]
        self.xrange = (x[0], x[-1])
        self.yrange = (y[0], y[-1])
        X, Y = np.meshgrid(x, y)
        self.coords = np.array([Y[subset_slice].flatten(), X[subset_slice].flatten()])
        self.S = interpolate.RectBivariateSpline(x, y, image, kx=kx, ky=ky)

    def __call__(self, x, y, dx=0, dy=0, grid=False, normalize=True):
        out = self.S(x, y, dx=dx, dy=dy, grid=grid)
        mask = (x >= self.xrange[0]) & (x <= self.xrange[1]) & (y >= self.yrange[0]) & (y <= self.yrange[1])
        noise_range = np.percentile(out, (0.0, 1.0))
        out[~mask] = np.random.uniform(noise_range[0], noise_range[1], np.sum(~mask))
        if normalize:
            return self.__normalize(out)
        return out

    def __normalize(self, a):
        mean = a.mean()
        return (a - mean) / np.sqrt(((a - mean)**2).sum())

    def warp(self, h, normalize=True):
        Wp = W(h)
        xi_3d = np.vstack((self.coords, np.ones(self.coords.shape[1])))
        xi_prime = Wp.dot(xi_3d)
        return self(xi_prime[0], xi_prime[1], normalize=normalize)

    def gradient(self):
        dx = self(self.coords[0], self.coords[1], dx=1, dy=0, grid=False, normalize=False)
        dy = self(self.coords[0], self.coords[1], dx=0, dy=1, grid=False, normalize=False)
        dxy = np.vstack((dx, dy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 1x2xN
        return dxy


def deform(xi: np.ndarray, spline: interpolate.RectBivariateSpline, p: np.ndarray) -> np.ndarray:
    """Deform a subset using a homography.
    TODO: Need to make it so that out-of-bounds points are replaced with noise.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        spline (interpolate.RectBivariateSpline): The biquintic B-spline of the subset.
        p (np.ndarray): The homography parameters. Shape is (8,)."""
    xi_prime = get_xi_prime(xi, p)
    return spline(xi_prime[0], xi_prime[1], grid=False)


def deform_image(image: np.ndarray,
                 p: np.ndarray,
                 PC: tuple | list | np.ndarray = None,
                 subset_slice: tuple = (slice(None), slice(None)),
                 kx: int = 2, ky: int = 2) -> np.ndarray:
    """Deform an image using a homography. Creates a bicubic spline of the image, deforms the coordinates of the pixels, and interpolates the deformed coordinates using the spline to get the deformed image.

    Args:
        image (np.ndarray): The image to be deformed.
        p (np.ndarray): The homography parameters. has shape (8,).
        PC (tuple): The pattern center. Default is None, which corresponds to the center of the image.
        subset_slice (tuple): The slice of the image to be deformed. Default is (slice(None), slice(None)).
        kx (int): The degree of the spline in the x direction. Default is 2.
        ky (int): The degree of the spline in the y direction. Default is 2.

    Returns:
        np.ndarray: The deformed image. Same shape as the input (unless subset_slice is used)."""
    if PC is None:
        PC = np.array([image.shape[1] / 2, image.shape[0] / 2])
    ii = np.arange(image.shape[0]) - PC[1]
    jj = np.arange(image.shape[1]) - PC[0]
    II, JJ = np.meshgrid(ii, jj, indexing="ij")
    xi = np.array([II[subset_slice].flatten(), JJ[subset_slice].flatten()])
    spline = interpolate.RectBivariateSpline(ii, jj, image, kx=kx, ky=ky)
    xi_prime = get_xi_prime(xi, p)
    tar_rot = spline(xi_prime[0], xi_prime[1], grid=False).reshape(image[subset_slice].shape)
    return tar_rot


def get_xi_prime(xi, p) -> np.ndarray:
    """Convert the subset coordinates to the deformed subset coordinates using the homography.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        p (np.ndarray): The homography parameters. Shape is (8,).

    Returns:
        np.ndarray: The deformed subset coordinates. Shape is (2, N)."""
    Wp = W(p)
    xi_3d = np.vstack((xi, np.ones(xi.shape[1])))
    xi_prime = Wp.dot(xi_3d)
    return xi_prime[:2] / xi_prime[2]


def W(p) -> np.ndarray:
    """Return the shape function matrix for the given homography parameters.
    Args:
        p (np.ndarray): The homography parameters.
    Returns:
        np.ndarray: The shape function matrix."""
    if p.ndim == 1:
        return np.concatenate((p, np.zeros(1)), axis=-1).reshape(3, 3) + np.eye(3)
    else:
        in_shape = p.shape[:-1]
        _0 = np.zeros(in_shape + (1,))
        if p.ndim == 2:
            return np.squeeze(np.concatenate((p, _0), axis=-1).reshape(in_shape + (3, 3,)) + np.eye(3)[None, ...])
        elif p.ndim == 3:
            return np.squeeze(np.concatenate((p, _0), axis=-1).reshape(in_shape + (3, 3,)) + np.eye(3)[None, None, ...])
        else:
            raise ValueError("p must be 1, 2, or 3 dimensions.")


### GPU functions

def W_vectorized_gpu(p) -> torch.Tensor:
    """Convert homographies into a shape function.
    Assumes p is a (B, 8) array."""
    in_shape = p.shape[:-1]
    _0 = torch.zeros(in_shape + (1,))
    return torch.cat((p, _0), dim=-1).reshape(in_shape + (3, 3,)) + torch.eye(3)[None, ...]


def get_xi_prime_vectorized_gpu(xi, p) -> np.ndarray:
    """Convert the subset coordinates to the deformed subset coordinates using the homography.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (H, W, 2).
        p (np.ndarray): The homography parameters. Shape is (B, 8).

    Returns:
        np.ndarray: The deformed subset coordinates. Shape is (B, H, W, 2)."""
    # Get dimensions
    batch_size = p.shape[0]
    shape = xi.shape[:-1]
    # Get shape function of homography
    Wp = W_vectorized_gpu(p)  # Bx3x3
    # Convert xi to 3D
    xi_3d = torch.cat((xi, torch.ones(*shape, 1)), dim=-1)  # HxWx3
    xi_3d = xi_3d.reshape(-1, 3).T  # 3xH*W
    xi_prime = torch.einsum("bij,jk->bik", Wp, xi_3d)  # Bx3x3 @ 3xH*W -> Bx3xH*W
    # xi_prime = torch.matmul(Wp, xi_3d)  # Bx3x3 @ 3xH*W -> Bx3xH*W
    xi_prime = xi_prime[:, :2, :] / xi_prime[:, -1:, :]  # Bx2xH*W
    xi_prime = torch.transpose(xi_prime, 1, 2).reshape(batch_size, *shape, 2)  # BxHxWx2
    return xi_prime