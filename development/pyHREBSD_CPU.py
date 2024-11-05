import numpy as np
from scipy import interpolate, linalg
from tqdm.auto import tqdm
import mpire


# Functions for running HREBSD


def W(p) -> np.ndarray:
    """Return the shape function matrix for the given homography parameters.
    Args:
        p (np.ndarray): The homography parameters.
    Returns:
        np.ndarray: The shape function matrix."""
    return np.concatenate((p, np.zeros(1)), axis=-1).reshape(3, 3) + np.eye(3)


def normalize(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    # img = (img - img.min()) / (img.max() - img.min())
    img_bar = img.mean()
    dimg_tilde = np.sqrt(((img - img_bar)**2).sum())
    return (img - img_bar) / dimg_tilde


def dp_norm(dp, xi) -> float:
    """Compute the norm of the deformation increment.
    This is essentially a modified form of a homography magnitude.

    Args:
        dp (np.ndarray): The deformation increment. Shape is (8,).
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        float: The norm of the deformation increment."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([xi1max, xi2max])
    dp_i0 = dp[0:2] * ximax
    dp_i1 = dp[3:5] * ximax
    dp_i2 = dp[6:8] * ximax
    out = np.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
    return out


def deform(xi: np.ndarray, spline: interpolate.RectBivariateSpline, p: np.ndarray) -> np.ndarray:
    """Deform a subset using a homography.
    TODO: Need to make it so that out-of-bounds points are replaced with noise.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        spline (interpolate.RectBivariateSpline): The biquintic B-spline of the subset.
        p (np.ndarray): The homography parameters. Shape is (8,)."""
    xi_prime = get_xi_prime(xi, p)
    out = spline(xi_prime[0], xi_prime[1], grid=False)
    return out


def get_xi_prime(xi, p) -> np.ndarray:
    """Convert the subset coordinates to the deformed subset coordinates using the homography.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        p (np.ndarray): The homography parameters. Shape is (8,).

    Returns:
        np.ndarray: The deformed subset coordinates. Shape is (2, N)."""
    Wp = W(p)
    xi_3d = np.vstack((xi, np.ones(xi.shape[1])))
    xi_prime = np.matmul(Wp, xi_3d)
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
    p = p0.copy()

    # Precompute cholesky decomposition of H
    c, lower = linalg.cho_factor(H)

    # Run the iterations
    num_iter = 0
    norms = []
    residuals = []
    while num_iter <= max_iter:
        # Warp the target subset
        num_iter += 1
        t_deformed = deform(xi, T_spline, p)

        # Compute the residuals
        e = r - normalize(t_deformed)
        residuals.append(np.abs(e).mean())

        # Copmute the gradient of the correlation criterion
        dC_IC_ZNSSD = 2 / dr_tilde * np.matmul(e, NablaR_dot_Jac.T)  # 8x1

        # Find the deformation incriment, delta_p, by solving the linear system
        # H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
        dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]

        # Update the parameters
        norm = dp_norm(dp, xi)
        Wp = W(p)
        Wdp = W(dp)
        Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
        p = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]

        # Store the update
        norms.append(norm)

        if norm < conv_tol:
            break

    if num_iter >= max_iter:
        print("Warning: Maximum number of iterations reached!")
    return p#, int(num_iter), float(residuals[-1])


# Vectorized versions


def W_vectorized(p) -> np.ndarray:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    in_shape = p.shape[:-1]
    _0 = np.zeros(in_shape + (1,))
    return np.concatenate((p, _0), axis=-1).reshape(in_shape + (3, 3,)) + np.eye(3)[None, ...]


def normalize_vectorized(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    img_bar = img.mean(axis=-1)[..., None]
    dimg_tilde = np.sqrt(((img - img_bar)**2).sum(axis=-1)[..., None])
    return (img - img_bar) / dimg_tilde


def dp_norm_vectorized(dp, xi) -> float:
    """Compute the norm of the delta p vector.
    Assumes dp is a (Nx8) array and xi is a (2,M) array."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([[xi1max, xi2max]])
    dp_i0 = np.square(dp[:, 0:2] * ximax).sum(axis=-1)
    dp_i1 = np.square(dp[:, 3:5] * ximax).sum(axis=-1)
    dp_i2 = np.square(dp[:, 6:8] * ximax).sum(axis=-1)
    out = np.sqrt(dp_i0 + dp_i1 + dp_i2 + np.square(dp[:, 2]) + np.square(dp[:, 5]))
    return out


def IC_GN_vectorized(p0, r, T, r_zmsv, NablaR_dot_Jac, H, xi, PC, conv_tol=1e-3, max_iter=50):
    # Precompute the target subset
    # print("Precomputing target subsets...")
    p = p0.copy()
    r = r.reshape(1, -1)  # 1 x M
    # with mpire.WorkerPool(n_jobs=2) as pool:
    #     T_splines = pool.map(target_precompute, [(T[i], PC) for i in range(len(T))])
    T_splines = [target_precompute(T[i], PC) for i in range(len(T))]

    # Convert the inputs to torch tensors
    # print("Converting inputs to torch tensors...")

    # Precompute cholesky decomposition of H
    # print("Precomputing Cholesky decomposition of H...")
    c, lower = linalg.cho_factor(H)

    # Run the iterations
    num_iter = np.zeros(T.shape[0], dtype=int)
    norms = np.zeros(T.shape[0])
    residuals = np.zeros(T.shape[0])
    active = np.ones(T.shape[0], dtype=bool)
    # N is the number of targets T.shape[0]
    # n is the number of targets that have converged (~active).sum()
    # M is the number of pixels in a target T.shape[1]
    while True:
        num_iter[active] += 1
        # Get the deformed target subsets
        # with mpire.WorkerPool(n_jobs=2) as pool:
        #     t_deformed = pool.map(deform, [(xi, T_splines[i], p[i]) for i in range(len(T_splines)) if active[i]])
        # t_deformed = t_deformed.reshape(active.sum(), r.shape[1])  # (N-n) x M
        t_deformed = []
        for i in range(len(T_splines)):
            if active[i]:
                t_deformed.append(deform(xi, T_splines[i], p[i]))
        t_deformed = np.vstack(t_deformed)  # (N-n) x M

        # Compute the residuals
        e = r - normalize_vectorized(t_deformed)  # (N-n) x M
        residuals[active] = np.abs(e).mean(axis=1)  # (N-n)

        # Compute the gradient of the correlation criterion
        dC_IC_ZNSSD = 2 / r_zmsv * np.einsum('ij,kj->ik', e, NablaR_dot_Jac)  # 8 x (N - n)
    
        # Find the deformation increment, dp, by solving the linear system
        # H @ dp = -dC_IC_ZNSSD using the Cholesky decomposition
        dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.T).T  # (N - n) x 8

        # Update the parameters
        norms_temp = dp_norm_vectorized(dp, xi)  # (N - n)
        Wp = W_vectorized(p[active])  # (N - n) x 3 x 3
        Wdp = W_vectorized(dp)  # (N - n) x 3 x 3
        Wpdp = Wp @ np.linalg.inv(Wdp)  # (N - n) x 3 x 3
        p_temp = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - np.eye(3)[None, ...])  # (N - n) x 3 x 3
        p_temp = p_temp.reshape(-1, 9)[:, :8]  # (N - n) x 8
        p[active] = p_temp
        norms[active] = norms_temp
        # Update the active targets
        active = norms > conv_tol
        # print(f"Iteration {num_iter.max()}: Number of active targets: {active.sum()}")
        if active.sum() == 0:
            break
        elif num_iter.max() >= max_iter:
            print("Warning: Maximum number of iterations reached!")
            break
    # Convert the outputs to numpy arrays
    return p#, num_iter, residuals
