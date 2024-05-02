import numpy as np
from scipy import interpolate, linalg
from tqdm.auto import tqdm

import rotations

def correct_geometry(H, PC, delPC) -> np.ndarray:
    """Apply projection geometry correction to a homography.

    Args:
        H (np.ndarray): The homography matrix.
        PC (np.ndarray): The pattern center.
        delPC (np.ndarray): The change in the pattern center.

    Returns:
        np.ndarray: The corrected homography."""
    x01, x02, DD = PC
    d1, d2, dDD = delPC
    alpha = (DD - dDD) / DD
    TS_inv = np.array([[1/alpha,       0, -(d1 + x01 * (alpha - 1)) / alpha],
                       [      0, 1/alpha, -(d2 + x02 * (alpha - 1)) / alpha],
                       [      0,       0,                                 1]])
    Wp = W(H)
    Wp_hat = TS_inv.dot(Wp)
    Wp_hat = Wp_hat / Wp_hat[2, 2]
    h = np.array([Wp_hat[0, 0] - 1, Wp_hat[0, 1],     Wp_hat[0, 2],
                  Wp_hat[1, 0],     Wp_hat[1, 1] - 1, Wp_hat[1, 2],
                  Wp_hat[2, 0],     Wp_hat[2, 1]])
    return h


def normalize(img) -> np.ndarray:
    """Zero-mean normalize an image with unit standard deviation.

    Args:
        img (np.ndarray): The image to normalize.

    Returns:
        np.ndarray: The normalized image."""
    # img = (img - img.min()) / (img.max() - img.min())
    img_bar = img.mean()
    dimg_tilde = np.sqrt(((img - img_bar)**2).sum())
    return (img - img_bar) / dimg_tilde


def jacobian(xi) -> np.ndarray:
    """Compute the jacobian of the shape function for a given subset.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        np.ndarray: The jacobian of the shape function. Shape is (2, 8, N)."""
    _1 = np.ones(xi.shape[1])
    _0 = np.zeros(xi.shape[1])
    out0 = np.array([[xi[0], xi[1],   _1,    _0,    _0,   _0,    -xi[0]**2, -xi[1]*xi[0]]])
    out1 = np.array([[   _0,    _0,   _0, xi[0], xi[1],   _1, -xi[0]*xi[1],    -xi[1]**2]])
    return np.vstack((out0, out1))


def W(p) -> np.ndarray:
    """Return the shape function matrix for the given homography parameters.
    Args:
        p (np.ndarray): The homography parameters.
    Returns:
        np.ndarray: The shape function matrix."""
    return np.array([[1 + p[0],     p[1], p[2]],
                     [    p[3], 1 + p[4], p[5]],
                     [    p[6],     p[7],    1]])


def deform(xi, spline, p) -> np.ndarray:
    """Deform a subset using a homography.

    Args:
        xi (np.ndarray): The subset coordinates. Shape is (2, N).
        spline (interpolate.RectBivariateSpline): The biquintic B-spline of the subset.
        p (np.ndarray): The homography parameters. Shape is (8,)."""
    xi_prime = get_xi_prime(xi, p)
    return spline(xi_prime[0], xi_prime[1], grid=False)


def reference_precompute(R, subset_slice) -> tuple:
    """Precompute arrays/values for the reference subset for the IC-GN algorithm.

    Args:
        R (np.ndarray): The reference subset.
        subset_slice (tuple): The slice of the subset to use.

    Returns:
        np.ndarray: The subset's zero-mean normalized intensities.
        float: The zero mean standard deviation of the subset's intensities.
        np.ndarray: The subset's intensity gradients.
        np.ndarray: The subset's Hessian matrix.
        np.ndarray: The subset's coordinates."""
    # Compute the subset's intensities. They are zero mean normalized.
    r = normalize(R)[subset_slice].flatten()

    # Get coordinates
    x = np.arange(R.shape[1]) - R.shape[1] / 2
    y = np.arange(R.shape[0]) - R.shape[0] / 2
    X, Y = np.meshgrid(x, y)
    xi = np.array([Y[subset_slice].flatten(), X[subset_slice].flatten()])

    # Compute the intensity gradients of the subset
    spline = interpolate.RectBivariateSpline(x, y, R, kx=5, ky=5)
    GRx = spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    GRy = spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    GR = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN
    r = spline(xi[0], xi[1], grid=False).flatten()
    r_zmsv = np.sqrt(((r - r.mean())**2).sum())
    r = (r - r.mean()) / r_zmsv

    # Get the reference intensities
    # r = spline(xi[0], xi[1], grid=False)
    # r = normalize(r)

    # Compute the jacobian of the shape function
    Jac = jacobian(xi)  # 2x8xN

    # Multiply the gradients by the jacobian
    NablaR_dot_Jac = np.einsum('ilk,ljk->ijk', GR, Jac)[0]  #1x8xN

    # Compute the Hessian
    H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)

    return r, r_zmsv, NablaR_dot_Jac, H, xi


def target_precompute(T) -> interpolate.RectBivariateSpline:
    """Precompute arrays/values for the target subset for the IC-GN algorithm.

    Args:
        T (np.ndarray): The target subset.

    Returns:
        interpolate.RectBivariateSpline: The biquintic B-spline of the target subset."""
    # Get coordinates
    x = np.arange(T.shape[1]) - T.shape[1] / 2
    y = np.arange(T.shape[0]) - T.shape[0] / 2

    # Compute the intensity gradients of the subset
    T_spline = interpolate.RectBivariateSpline(x.flatten(), y.flatten(), normalize(T), kx=5, ky=5)

    return T_spline


def dp_norm(dp, xi) -> float:
    """Compute the norm of the deformation increment.

    Args:
        dp (np.ndarray): The deformation increment. Shape is (8,).
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        float: The norm of the deformation increment."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([xi1max, xi2max])
    dp_i0 = np.array([dp[0], dp[1]]) * ximax
    dp_i1 = np.array([dp[3], dp[4]]) * ximax
    dp_i2 = np.array([dp[6], dp[7]]) * ximax
    out = np.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
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
    xi_prime = Wp.dot(xi_3d)
    return xi_prime[:2] / xi_prime[2]


def get_homography(R, T, subset_slice=None, conv_tol=1e-5, max_iter=50, p0=None) -> np.ndarray:
    """Run the inverse compositional Gauss-Newton algorithm on a set of EBSD patterns.

    Args:
        R (np.ndarray): The reference pattern.
        T (np.ndarray): The target pattern(s). 2D (single pattern), 3D (multiple patterns), or 4D (grid of patterns) accepted.
        subset_slice (tuple): The slice of the subset to use. Default is None, which uses the entire subset.
        conv_tol (float): The convergence tolerance. Default is 1e-5.
        max_iter (int): The maximum number of iterations. Default is 50.
        p0 (np.ndarray): The initial guess for the homography parameters. Must be the same shape as T.
                         Default is None, which corresponds to zero-valued homographies.

    Returns:
        np.ndarray: The homography parameters. Will have the same shape as T.

    TODO:
        Add pattern center correction.
        Explore multiprocessing or GPU acceleration.
    """
    # Setup the target patterns so that they are 3D
    if T.ndim == 2:
        T = T[None]  # (1, H, W)
    if p0 is not None and p0.ndim == 1:
        p0 = p0[None]

    # Check the shape of the target patterns and the initial guesses
    if p0 is not None:
        if p0.shape != T.shape[:-2] + (8,):
            raise ValueError("The initial guess must have the same shape as the target patterns.")
    else:
        p0 = np.zeros((T.shape[:-2]) + (8,), dtype=np.float32)

    # Reshape the target patterns to 3D if they are 4D
    if T.ndim == 4:
        out_shape = T.shape[:2]
        p0 = p0.reshape(T.shape[0] * T.shape[1], 8)
        T = T.reshape(T.shape[0] * T.shape[1], T.shape[2], T.shape[3])
    else:
        out_shape = T.shape[:1]

    # Set up the subset slice
    if subset_slice is None:
        subset_slice = (slice(None), slice(None))

    # Precompute values for the reference subset
    r, dr_tilde, NablaR_dot_Jac, H, xi = reference_precompute(R, subset_slice)
    p_out = np.zeros_like(p0)

    # Start loop over targets
    for i in tqdm(range(T.shape[0]), desc="IC-GN", leave=False, unit="targets"):
        # Precompute the target subset
        p = p0[i]
        T_spline = target_precompute(T[i])

        # Run the iterations
        num_iter = 0
        while True:
            # Warp the target subset
            num_iter += 1
            t_deformed = deform(xi, T_spline, p)
            t_deformed = normalize(t_deformed)

            # Compute the residuals
            e = r - t_deformed

            # Copmute the gradient of the correlation criterion
            dC_IC_ZNSSD = 2 / dr_tilde * np.matmul(e, NablaR_dot_Jac.T)  # 8x1

            # Find the deformation incriment, delta_p, by solving the linear system H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
            c, lower = linalg.cho_factor(H)
            dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]

            # Update the homography
            norm = dp_norm(dp, xi)
            Wp = W(p)
            Wp = Wp.dot(np.linalg.inv(W(dp)))
            Wp = Wp / Wp[2, 2]
            p = np.array([Wp[0, 0] - 1, Wp[0, 1], Wp[0, 2],
                          Wp[1, 0], Wp[1, 1] - 1, Wp[1, 2],
                          Wp[2, 0], Wp[2, 1]])

            # Check for convergence
            if norm < conv_tol or num_iter < max_iter:
                break

        # Check if the maximum number of iterations was reached and print a warning if so
        if num_iter == max_iter:
            print(f"Warning: Maximum number of iterations reached for target {i}")

        # Store the homography
        p_out[i] = p

    # Reshape to match the input
    p_out = np.squeeze(p_out.reshape(out_shape + (8,)))
    return p_out


def homography_to_elastic_deformation(H, PC):
    """Calculate the deviatoric deformation gradient from a homography using the projection geometry (pattern center).
    Note that the deformation gradient is insensitive to hydrostatic dilation.

    Args:
        H (np.ndarray): The homography matrix.
        PC (np.ndarray): The pattern center.

    Returns:
        np.ndarray: The deviatoric deformation gradient."""
    # Reshape the homography if necessary
    if H.ndim == 1:
        H.reshape(1, 8)

    # Extract the data from the inputs (not necessary, but easier to read)
    x01, x02, DD = PC
    h11, h12, h13, h21, h22, h23, h31, h32 = H[..., 0], H[..., 1], H[..., 2], H[..., 3], H[..., 4], H[..., 5], H[..., 6], H[..., 7]

    # Calculate the deformation gradient
    beta0 = 1 - h31 * x01 - h32 * x02
    Fe11 = 1 + h11 + h31 * x01
    Fe12 = h12 + h32 * x01
    Fe13 = (h13 - h11*x01 - h12*x02 + x01*(beta0 - 1))/DD
    Fe21 = h21 + h31 * x02
    Fe22 = 1 + h22 + h32 * x02
    Fe23 = (h23 - h21*x01 - h22*x02 + x02*(beta0 - 1))/DD
    Fe31 = DD * h31
    Fe32 = DD * h32
    Fe33 = beta0
    Fe = np.array([[Fe11, Fe12, Fe13], [Fe21, Fe22, Fe23], [Fe31, Fe32, Fe33]]) / beta0

    # Reshape the output if necessary
    if Fe.ndim == 4:
        Fe = np.moveaxis(Fe, (0, 1, 2, 3), (2, 3, 0, 1))
    elif Fe.ndim == 3:
        Fe = np.squeeze(np.moveaxis(Fe, (0, 1, 2), (2, 0, 1)))

    return Fe


def deformation_to_strain(Fe: np.ndarray) -> tuple:
    """Calculate the elastic strain tensor from the deformation gradient.
    Also calculates the lattice rotation matrix.

    Args:
        Fe (np.ndarray): The deformation gradient.

    Returns:
        epsilon (np.ndarray): The elastic strain tensor
        omega (np.ndarray): The lattice rotation matrix."""
    if Fe.ndim == 4:
        I = np.eye(3)[None, None, ...]
    elif Fe.ndim == 3:
        I = np.squeeze(np.eye(3)[None, ...])
    # Calculate the small strain tensor
    # Use small strain theory to decompose the deformation gradient into the elastic strain tensor and the rotation matrix
    d = Fe - I
    if Fe.ndim == 3:
        dT = d.transpose(0, 2, 1)
    elif Fe.ndim == 4:
        dT = d.transpose(0, 1, 3, 2)
    epsilon = 0.5 * (d + dT)

    # Calculate the rotation tensor
    # Use the polar decomposition of the deformation gradient to decompose it into the rotation matrix and the stretch tensor
    W, S, V = np.linalg.svd(Fe, full_matrices=True)
    omega_finite = np.matmul(W, V)
    # Sigma = np.einsum('...i,ij->...ij', S, np.eye(3))
    # epsilon = np.matmul(W, np.matmul(Sigma, W.transpose(0, 1, 3, 2))) - I

    # Convert finite rotation matrix to a lattice rotation matrix
    v = rotations.om2ax(omega_finite)
    rotation_vector = v[..., :3] * v[..., 3][..., None]
    omega = np.zeros_like(omega_finite)
    omega[..., 0, 1] = -rotation_vector[..., 2]
    omega[..., 0, 2] = rotation_vector[..., 1]
    omega[..., 1, 2] = rotation_vector[..., 0]
    omega[..., 1, 0] = -omega[..., 0, 1]
    omega[..., 2, 0] = -omega[..., 0, 2]
    omega[..., 2, 1] = -omega[..., 1, 2]

    # Convert to sample frame rotation
    if Fe.ndim == 3:
        omega = np.transpose(omega, (0, 2, 1))
    elif Fe.ndim == 4:
        omega = np.transpose(omega, (0, 1, 3, 2))

    return epsilon, omega