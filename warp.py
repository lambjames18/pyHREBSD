import numpy as np
from scipy import interpolate, linalg



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
    return np.concatenate((p, np.zeros(1)), axis=-1).reshape(3, 3) + np.eye(3)
