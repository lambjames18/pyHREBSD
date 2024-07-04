import numpy as np
from scipy import interpolate
from tqdm.auto import tqdm
import torch
import mpire
import bspline_gpu as gpu_warp

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Functions for running HREBSD


def W_gpu(p) -> np.ndarray:
    """Return the shape function matrix for the given homography parameters.
    Args:
        p (np.ndarray): The homography parameters.
    Returns:
        np.ndarray: The shape function matrix."""
    return (torch.cat((p, torch.zeros(1, device=device)), dim=-1).reshape(3, 3) + torch.eye(3, device=device)).float()


def normalize_gpu(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    # img = (img - img.min()) / (img.max() - img.min())
    img_bar = img.mean()
    dimg_tilde = torch.sqrt(((img - img_bar)**2).sum())
    return (img - img_bar) / dimg_tilde


def dp_norm_gpu(dp, xi) -> float:
    """Compute the norm of the deformation increment.
    This is essentially a modified form of a homography magnitude.

    Args:
        dp (np.ndarray): The deformation increment. Shape is (8,).
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        float: The norm of the deformation increment."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = torch.tensor([xi1max, xi2max], device=device).float()
    dp_i0 = dp[0:2] * ximax
    dp_i1 = dp[3:5] * ximax
    dp_i2 = dp[6:8] * ximax
    out = torch.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
    return out


def deform(xi: np.ndarray, spline: interpolate.RectBivariateSpline, p: np.ndarray) -> np.ndarray:
    """Deform a subset using a homography.
    TODO: Need to make it so that out-of-bounds points are replaced with noise.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        spline (interpolate.RectBivariateSpline): The biquintic B-spline of the subset.
        p (np.ndarray): The homography parameters. Shape is (8,)."""
    xi_prime = get_xi_prime_gpu(xi, p).detach().cpu().numpy()
    out = spline(xi_prime[0], xi_prime[1], grid=False)
    return torch.tensor(out, device=device).float()


def get_xi_prime_gpu(xi, p) -> np.ndarray:
    """Convert the subset coordinates to the deformed subset coordinates using the homography.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        p (np.ndarray): The homography parameters. Shape is (8,).

    Returns:
        np.ndarray: The deformed subset coordinates. Shape is (2, N)."""
    Wp = W_gpu(p)
    xi_3d = torch.vstack((xi, torch.ones(xi.shape[1], device=device))).float()
    xi_prime = torch.matmul(Wp, xi_3d)
    return xi_prime[:2] / xi_prime[2]


def target_precompute(T: np.ndarray, PC: np.ndarray) -> interpolate.RectBivariateSpline:
    """Precompute arrays/values for the target subset for the IC-GN algorithm.

    Args:
        T (np.ndarray): The target subset.
        xi (np.ndarray): The subset's coordinates. Shape is (2, N). Default is None, leading to the coordinates being calculated.

    Returns:
        interpolate.RectBivariateSpline: The biquintic B-spline of the target subset."""
    # Get coordinates
    x = np.arange(T.shape[1]) - PC[0]
    y = np.arange(T.shape[0]) - PC[1]

    # Compute the intensity gradients of the subset
    T_spline = interpolate.RectBivariateSpline(x, y, T, kx=5, ky=5)

    return T_spline


def IC_GN(p0, r, T, dr_tilde, NablaR_dot_Jac, H, xi, PC, conv_tol=1e-3, max_iter=50) -> np.ndarray:
    # Precompute the target subset
    T_spline = target_precompute(T, PC)

    # Convert the inputs to torch tensors
    r = torch.tensor(r, device=device)
    T = torch.tensor(T, device=device)
    NablaR_dot_Jac = torch.tensor(NablaR_dot_Jac, device=device)
    H = torch.tensor(H, device=device)
    xi = torch.tensor(xi, device=device)
    p = torch.tensor(p0, device=device)

    # Precompute cholesky decomposition of H
    L = torch.linalg.cholesky(H)

    # Run the iterations
    num_iter = 0
    norms = []
    residuals = []
    while num_iter <= max_iter:
        # Warp the target subset
        num_iter += 1
        t_deformed = deform(xi, T_spline, p)

        # Compute the residuals
        e = r - normalize_gpu(t_deformed)
        residuals.append(torch.abs(e).mean())

        # Copmute the gradient of the correlation criterion
        dC_IC_ZNSSD = 2 / dr_tilde * torch.matmul(e, NablaR_dot_Jac.T)  # 8x1

        # Find the deformation incriment, delta_p, by solving the linear system
        # H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
        dp = torch.cholesky_solve(-dC_IC_ZNSSD.reshape(-1, 1), L)[:, 0]

        # Update the parameters
        norm = dp_norm_gpu(dp, xi)
        Wp = W_gpu(p)
        Wdp = W_gpu(dp)
        Wpdp = torch.matmul(Wp, torch.linalg.inv(Wdp))
        p = ((Wpdp / Wpdp[2, 2]) - torch.eye(3, device=device)).reshape(9)[:8]

        # Store the update
        norms.append(norm)

        if norm < conv_tol:
            break

    if num_iter >= max_iter:
        print("Warning: Maximum number of iterations reached!")
    p = p.detach().cpu().numpy()
    return p#, int(num_iter), float(residuals[-1])


# Vectorized versions


def W_vectorized_gpu(p) -> torch.Tensor:
    """Convert homographies into a shape function.
    Assumes p is a (B, 8) array."""
    in_shape = p.shape[:-1]
    _0 = torch.zeros(in_shape + (1,), device=device)
    return torch.cat((p, _0), dim=-1).reshape(in_shape + (3, 3,)) + torch.eye(3, device=device)[None, ...]


def normalize_vectorized_gpu(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    img_bar = img.mean(axis=-1)[..., None]
    dimg_tilde = torch.sqrt(((img - img_bar)**2).sum(axis=-1)[..., None])
    return (img - img_bar) / dimg_tilde


def dp_norm_vectorized_gpu(dp, xi) -> float:
    """Compute the norm of the delta p vector.
    Assumes dp is a (B, 8) array and xi is a (*, 2) array."""
    xi1max = xi[..., 0].max()
    xi2max = xi[..., 1].max()
    ximax = torch.tensor([[xi1max, xi2max]], device=device)  # Bx2
    dp_i0 = torch.square(dp[:, 0:2] * ximax).sum(axis=-1)
    dp_i1 = torch.square(dp[:, 3:5] * ximax).sum(axis=-1)
    dp_i2 = torch.square(dp[:, 6:8] * ximax).sum(axis=-1)
    out = torch.sqrt(dp_i0 + dp_i1 + dp_i2 + torch.square(dp[:, 2]) + torch.square(dp[:, 5]))
    return out


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
    xi_3d = torch.cat((xi, torch.ones(*shape, 1, device=device)), dim=-1)  # HxWx3
    xi_3d = xi_3d.reshape(-1, 3).T  # 3xH*W
    xi_prime = torch.matmul(Wp, xi_3d)  # Bx3x3 @ 3xH*W -> Bx3xH*W
    xi_prime = xi_prime[:, :2, :] / xi_prime[:, 1:, :]  # Bx2xH*W
    xi_prime = torch.transpose(xi_prime, 1, 2).reshape(batch_size, *shape, 2)  # BxHxWx2
    return xi_prime


def IC_GN_vectorized(p0, r, T, r_zmsv, NablaR_dot_Jac, H, xi, PC, conv_tol=1e-3, max_iter=50):
    # Get shape variables
    print(r.shape, T.shape, xi.shape)
    shape = np.sqrt(r.shape[0]).astype(int)
    batch = T.shape[0]
    # Convert the inputs to torch tensors
    print("Converting inputs to torch tensors...")
    r = torch.tensor(r, device=device).float().reshape(1, 1, shape, shape)  # 1x1xHxW
    T = torch.tensor(T, device=device).float().reshape(batch, 1, shape, shape)  # Bx1xHxW
    p = torch.tensor(p0, device=device).float()  # Bx8
    NablaR_dot_Jac = torch.tensor(NablaR_dot_Jac, device=device).float()  # 8x8
    H = torch.tensor(H, device=device).float()  # 8x8
    xi = torch.tensor(xi.T, device=device).float().reshape(shape, shape, 2)  # 2xM
    print(r.shape, T.shape, xi.shape)

    # Precompute cholesky decomposition of H
    print("Precomputing Cholesky decomposition of H...")
    L = torch.linalg.cholesky(H)

    # Precompute the spline and bounds
    bound_fn = gpu_warp.make_bound([0, 0])
    spline_fn = gpu_warp.make_spline([5, 5])

    # Run the iterations
    num_iter = torch.zeros(T.shape[0], dtype=int, device=device)
    norms = torch.zeros(T.shape[0], device=device)
    residuals = torch.zeros(T.shape[0], device=device)
    active = torch.ones(T.shape[0], dtype=bool, device=device)
    print(f"Number of targets (B): {T.shape[0]}")
    print(f"Height (H): {shape}")
    print(f"Width (W): {shape}")
    while True:
        num_iter += 1
        # Rotate the coordinates and warp the target subset
        xi_prime = get_xi_prime_vectorized_gpu(xi, p)  # BxHxWx2
        t_deformed = gpu_warp.pull(T, xi_prime, bound_fn, spline_fn, extrapolate=False)  # Bx1xHxW
        # Compute the residuals
        e = (r - normalize_vectorized_gpu(t_deformed)).reshape(batch, -1)  # BxM
        residuals = torch.abs(e).mean(dim=1)  # B
        # Compute the gradient of the correlation criterion
        dC_IC_ZNSSD = 2 / r_zmsv * torch.einsum('ij,kj->ik', e, NablaR_dot_Jac).T  # Bx8
        # Find the deformation increment
        dp = torch.cholesky_solve(-dC_IC_ZNSSD, L).T  # Bx8, 8x8 -> Bx8
        # Update the parameters
        Wp = W_vectorized_gpu(p[active])  # Bx3x3
        Wdp = W_vectorized_gpu(dp)  # Bx3x3
        Wpdp = Wp @ torch.linalg.inv(Wdp)  # Bx3x3
        p = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - torch.eye(3, device=device)[None, ...])  # Bx3x3
        p = p.reshape(-1, 9)[:, :8]  # Bx8
        # Compute the norm of the deformation increment and check for convergence
        norms = dp_norm_vectorized_gpu(dp, xi)  # B
        if norms.max() < conv_tol:
            break
        elif num_iter.max() >= max_iter:
            print("Warning: Maximum number of iterations reached!")
            break
    # Convert the outputs to numpy arrays
    p = p.detach().cpu().numpy()
    num_iter = num_iter.detach().cpu().numpy().astype(int)
    residuals = residuals.detach().cpu().numpy().astype(float)
    return p, num_iter, residuals
