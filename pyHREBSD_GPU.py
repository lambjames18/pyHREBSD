import numpy as np
from scipy import interpolate
from tqdm.auto import tqdm
import torch
import mpire

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
    return torch.cat((p, torch.zeros(1, device=device)), dim=-1).reshape(3, 3) + torch.eye(3, device=device)


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
    ximax = torch.tensor([xi1max, xi2max], device=device)
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
    xi_3d = torch.vstack((xi, torch.ones(xi.shape[1])))
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
    return p, int(num_iter), float(residuals[-1])


# Vectorized versions


def W_alt_gpu(p) -> np.ndarray:
    """Return the shape function matrix for the given homography parameters.
    Args:
        p (np.ndarray): The homography parameters.
    Returns:
        np.ndarray: The shape function matrix."""
    return torch.cat((p, torch.zeros(1)), dim=-1).reshape(3, 3) + torch.eye(3)


def W_vectorized_gpu(p) -> torch.Tensor:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
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
    Assumes dp is a (Nx8) array and xi is a (2,M) array."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = torch.tensor([[xi1max, xi2max]], device=device)
    dp_i0 = torch.square(dp[:, 0:2] * ximax).sum(axis=-1)
    dp_i1 = torch.square(dp[:, 3:5] * ximax).sum(axis=-1)
    dp_i2 = torch.square(dp[:, 6:8] * ximax).sum(axis=-1)
    out = torch.sqrt(dp_i0 + dp_i1 + dp_i2 + torch.square(dp[:, 2]) + torch.square(dp[:, 5]))
    return out


def deform_alt(xi: np.ndarray, spline: interpolate.RectBivariateSpline, p: np.ndarray) -> np.ndarray:
    """Deform a subset using a homography.
    TODO: Need to make it so that out-of-bounds points are replaced with noise.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        spline (interpolate.RectBivariateSpline): The biquintic B-spline of the subset.
        p (np.ndarray): The homography parameters. Shape is (8,)."""
    xi_prime = get_xi_prime_alt_gpu(xi, p).detach().cpu().numpy()
    out = spline(xi_prime[0], xi_prime[1], grid=False)
    return torch.tensor(out)


def get_xi_prime_alt_gpu(xi, p) -> np.ndarray:
    """Convert the subset coordinates to the deformed subset coordinates using the homography.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        p (np.ndarray): The homography parameters. Shape is (8,).

    Returns:
        np.ndarray: The deformed subset coordinates. Shape is (2, N)."""
    Wp = W_alt_gpu(p)
    xi_3d = torch.vstack((xi, torch.ones(xi.shape[1])))
    xi_prime = torch.matmul(Wp, xi_3d)
    return xi_prime[:2] / xi_prime[2]


def IC_GN_vectorized(p0, r, T, r_zmsv, NablaR_dot_Jac, H, xi, PC, conv_tol=1e-3, max_iter=50):
    # Precompute the target subset
    print("Precomputing target subsets...")
    with mpire.WorkerPool(n_jobs=mpire.cpu_count() // 2) as pool:
        T_splines = pool.map(target_precompute, [(T[i], PC) for i in range(len(T))])
    xi_3d = np.vstack((np.ones(xi.shape[1]), xi[0], xi[1])).T
    T_spline_3D = interpolate.RBFInterpolator(xi_3d, T.T, kernel="quintic", neighbors=25)

    # Convert the inputs to torch tensors
    print("Converting inputs to torch tensors...")
    r = torch.tensor(r, device=device).float()  # M
    T = torch.tensor(T, device=device).float()  # NxM
    p = torch.tensor(p0, device=device).float()  # Nx8
    NablaR_dot_Jac = torch.tensor(NablaR_dot_Jac, device=device).float()  # 8x8
    H = torch.tensor(H, device=device).float()  # 8x8
    xi = torch.tensor(xi, device=device).float()  # 2xM

    # Precompute cholesky decomposition of H
    print("Precomputing Cholesky decomposition of H...")
    L = torch.linalg.cholesky(H)

    # Run the iterations
    num_iter = torch.zeros(T.shape[0], dtype=int, device=device)
    norms = torch.zeros(T.shape[0], device=device)
    residuals = torch.zeros(T.shape[0], device=device)
    active = torch.ones(T.shape[0], dtype=bool, device=device)
    # N is the number of targets
    # n is the number of targets that have converged
    # M is the number of pixels in a target
    print(f"Number of targets (N): {T.shape[0]}")
    print(f"Number of pixels in a target (M): {r.shape[0]}")
    while True:
        num_iter[active] += 1
        xi = xi.cpu()
        p = p.cpu()
        # with mpire.WorkerPool(n_jobs=mpire.cpu_count() // 2) as pool:
        #     t_deformed = pool.map(deform_alt, [(xi, T_splines[i], p[i]) for i in range(len(T_splines)) if active[i]])
        xi = xi.to(device)
        p = p.to(device)
        # t_deformed = []
        # for i in range(len(T_splines)):
        #     if active[i]:
        #         t_deformed.append(deform(xi, T_splines[i], p[i]))
        t_deformed = torch.stack(t_deformed).float().to(device)  # (N-n) x M
        e = r - normalize_vectorized_gpu(t_deformed)  # (N-n) x M
        residuals[active] = torch.abs(e).mean(dim=1)  # (N-n)
        dC_IC_ZNSSD = 2 / r_zmsv * torch.einsum('ij,kj->ik', e, NablaR_dot_Jac)  # 8 x (N - n)
        dp = torch.cholesky_solve(-dC_IC_ZNSSD.T, L).T  # (N - n) x 8
        norms_temp = dp_norm_vectorized_gpu(dp, xi)  # (N - n)
        Wp = W_vectorized_gpu(p[active])  # (N - n) x 3 x 3
        Wdp = W_vectorized_gpu(dp)  # (N - n) x 3 x 3
        Wpdp = Wp @ torch.linalg.inv(Wdp)  # (N - n) x 3 x 3
        p_temp = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - torch.eye(3, device=device)[None, ...])  # (N - n) x 3 x 3
        p_temp = p_temp.reshape(-1, 9)[:, :8]  # (N - n) x 8
        p[active] = p_temp
        norms[active] = norms_temp
        # Update the active targets
        active = norms > conv_tol
        print(f"Iteration {num_iter.max()}: Number of active targets: {active.sum()}")
        if active.sum() == 0:
            break
        elif num_iter.max() >= max_iter:
            print("Warning: Maximum number of iterations reached!")
            break
    # Convert the outputs to numpy arrays
    p = p.detach().cpu().numpy()
    num_iter = num_iter.detach().cpu().numpy().astype(int)
    residuals = residuals.detach().cpu().numpy().astype(float)
    return p#, num_iter, residuals
