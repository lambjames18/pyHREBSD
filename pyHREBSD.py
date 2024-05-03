import numpy as np
from scipy import interpolate, linalg, signal, ndimage
from tqdm.auto import tqdm
import mpire

import rotations


### Functions for running HREBSD ###


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


### Functions for the initial guesses ###


def Tukey_Hanning_window(sig, alpha=0.4, return_window=False):
    """Applies a Tukey-Hanning window to the input signal.
    Args:
        sig (np.ndarray): The input signal. The last two dimensions should be the image to be windowed. (N, M, H, W) or (N, H, W) or (H, W) for example.
        alpha (float): The alpha parameter of the Tukey-Hanning window.
    Returns:
        np.ndarray: The windowed signal."""
    if sig.ndim == 1:
        window = signal.windows.tukey(sig.shape[-1], alpha=alpha)
    else:
        window_row = signal.windows.tukey(sig.shape[-2], alpha=alpha)
        window_col = signal.windows.tukey(sig.shape[-1], alpha=alpha)
        window = np.outer(window_row, window_col)
        while sig.ndim > window.ndim:
            window = window[None, :]
    if return_window:
        return sig * window, window
    else:
        return sig * window


def window_and_normalize(images, alpha=0.4):
    """Applies a Tukey-Hanning window and normalizes the input images.
    Args:
        images (np.ndarray): The input images. The last two dimensions should be the image to be windowed. (N, M, H, W) or (N, H, W) or (H, W) for example.
        alpha (float): The alpha parameter of the Tukey-Hanning window.
    Returns:
        np.ndarray: The windowed and normalized images."""
    # Get axis to operate on
    if images.ndim >= 2:
        axis = (-2, -1)
    else:
        axis = -1
    # Apply the Tukey-Hanning window
    windowed, window = Tukey_Hanning_window(images, alpha, return_window=True)
    # Get the normalizing factors   
    image_bar = images.mean(axis=axis)
    windowed_bar = (images * windowed).mean(axis=axis)
    bar = windowed_bar / image_bar
    del windowed, image_bar, windowed_bar
    while bar.ndim < images.ndim:
        bar = bar[..., None]
    # Window and normalize the image
    new_normalized_windowed = (images - bar) * window
    del window, bar
    variance = (new_normalized_windowed**2).sum(axis=axis) / (np.prod(images.shape[-2:]) - 1)
    while variance.ndim < images.ndim:
        variance = variance[..., None]
    out = new_normalized_windowed / np.sqrt(variance)
    return out


def FMT(image, X, Y, x, y):
    """Fourier-Mellin Transform of an image in which polar resampling is applied first.
    Args:
        image (np.ndarray): The input image of shape (2**n, 2**n)
        X (np.ndarray): The x-coordinates of the input image. Should correspond to the x coordinate of the image.
        Y (np.ndarray): The y-coordinates of the input image. Should correspond to the y coordinate of the image.
        x (np.ndarray): The x-coordinates of the output image. Should correspond to the x coordinates of the polar image.
        y (np.ndarray): The y-coordinates of the output image. Should correspond to the y coordinates of the polar image.
    Returns:
        np.ndarray: The signal of the Fourier-Mellin Transform. (1D array of length 2**n)"""
    spline = interpolate.RectBivariateSpline(X, Y, image.real, kx=2, ky=2)
    image_polar = np.abs(spline(x, y, grid=False).reshape(image.shape))
    sig = window_and_normalize(image_polar.mean(axis=1))
    return sig, image_polar


def _get_gcc_guesses(ref: np.ndarray, targets: np.ndarray):
    """Perform a global cross-correlation initial guess for the homographies of the targets.
    Rotation is determined using a Fourier-Mellin Transform.
    Translation is determined using a 2D cross-correlation.
    The images are cropped to 128x128 to speed up the process.
    Args:
        ref (np.ndarray): The reference pattern. (H, W)
        targets (np.ndarray): The target patterns. (M, N, H, W) or (N, H, W) or (H, W) for example.
        progress (int): The index of the target in the targets array. Used for logging during multiprocessing.
    Returns:
        np.ndarray: The homographies of the targets. (M, N, 8) or (N, 8) or (8) for example."""
    # Create the subset slice
    c = np.array(ref.shape) // 2
    subset_slice = (slice(c[0] - 64, c[0] + 64), slice(c[1] - 64, c[1] + 64))

    # Window and normalize the reference and targets
    ref = window_and_normalize(ref[subset_slice])
    subset_slice = (slice(None),) + subset_slice
    targets = window_and_normalize(targets[subset_slice])

    # Get the dimensions of the image
    height, width = ref.shape
    n = np.log2(height)

    # Create a mesh grid of log-polar coordinates
    theta = np.linspace(0, np.pi, int(2**n), endpoint=False)
    radius = np.linspace(0, height / 2, int(2**n + 1), endpoint=False)[1:]
    radius_grid, theta_grid = np.meshgrid(radius, theta, indexing='xy')
    radius_grid = radius_grid.flatten()
    theta_grid = theta_grid.flatten()

    # Convert log-polar coordinates to Cartesian coordinates
    x = 2**(n-1) + radius_grid * np.cos(theta_grid)
    y = 2**(n-1) - radius_grid * np.sin(theta_grid)

    # Create a mesh grid of Cartesian coordinates
    X = np.arange(width)
    Y = np.arange(height)

    # FFT the reference and get the signal
    # ref_fft = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(ref)) + 1))
    ref_fft = np.fft.fftshift(np.fft.fft2(ref))
    ref_FMT, ref_polar = FMT(ref_fft, X, Y, x, y)

    # Create arrays to store the measurements
    measurements = np.zeros((len(targets), 3), dtype=np.float32)

    # Loop through the targets
    for i in range(len(targets)):
        tar = targets[i]
        # Do the angle search first
        tar_fft = np.fft.fftshift(np.fft.fft2(tar))
        tar_FMT, tar_polar = FMT(tar_fft, X, Y, x, y)
        cc = signal.correlate(ref_FMT, tar_FMT, mode='same', method="fft")
        theta = - (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)
        # Apply the rotation
        tar_rot = ndimage.rotate(tar, np.degrees(theta), reshape=False)
        # Do the translation search
        cc = signal.correlate2d(ref, tar_rot, mode='same').real
        shift = np.unravel_index(np.argmax(cc), cc.shape) - np.array(cc.shape) / 2
        # Store the homography
        measurements[i] = np.array([shift[0], shift[1], theta])

    # Convert the measurements to homographies
    _0 = np.zeros(measurements.shape[0])
    _c = np.cos(measurements[:, 2])
    _s = np.sin(measurements[:, 2])
    _x = measurements[:, 0] * np.ones(measurements.shape[0])
    _y = measurements[:, 1] * np.ones(measurements.shape[0])
    homographies = np.array([_c - 1, -_s, _x*_c-_y*_s, _s, _c - 1, _x*_s+_y*_c, _0, _0]).T
    # if progress is not None:
    #     print(f"Progress: {round(progress * 100, 2)}%" + " "*10, end="\r", flush=True)
    return homographies


def get_initial_guess(ref, targets, split_size=7):
    # Check inputs
    targets = np.asarray(targets)

    # Get the guesses
    if targets.ndim == 2:
        # Case where we have a single target
        return np.squeeze(_get_gcc_guesses(ref, targets.reshape(1, *targets.shape)))
    elif targets.ndim == 3 and targets.shape[0] < 100:
        # Case where we have a 1D array of targets but not enough to parallelize
        return _get_gcc_guesses(ref, targets)
    elif targets.ndim == 4 and targets.size < 100:
        # Case where we have a 2D array of targets but not enough to parallelize
        shape = targets.shape[:2]
        return _get_gcc_guesses(ref, targets.reshape(-1, targets.shape[2], targets.shape[3])).reshape(shape + (8,))
    elif targets.ndim == 3 and targets.shape[0] >= 100:
        # Case where we have a 1D array of targets and enough to parallelize
        print("There are enough targets to parallelize. Starting pool.")
        N = mp.cpu_count() // 2
        splits = np.array_split(targets, split_size)
        with mp.Pool(N) as pool:
            results = pool.starmap(_get_gcc_guesses, [(ref, split, i/len(splits)) for i, split in enumerate(splits)])
        return np.concatenate(results)
    else:
        # Case where we have a 2D array of targets and enough to parallelize
        print("There are enough targets to parallelize. Starting pool.")
        shape = targets.shape[:2]
        N = mpire.cpu_count() // 2
        splits = np.array_split(targets.reshape(-1, targets.shape[2], targets.shape[3]), np.prod(shape[:2]) // split_size)
        with mpire.WorkerPool(n_jobs=N) as pool:
            results = pool.map(_get_gcc_guesses, [(ref, split) for split in splits], progress_bar=True)
        # with mp.Pool(N) as pool:
        #     results = pool.starmap(_get_gcc_guesses, [(ref, split, i/len(splits)) for i, split in enumerate(splits)])
        return np.concatenate(results).reshape(shape + (8,))


if __name__ == "__main__":
    import time
    import utilities

    up2 = "E:/cells/CoNi90-OrthoCells.up2"
    ang = "E:/cells/CoNi90-OrthoCells.ang"
    save_name = "CoNi90-OrthoCells_R112-96_100x100_DoG1-10_Equalize"
    pixel_size = 13.0
    Nxy = (2048, 2048)
    point = (112, 96)
    size = 100
    DoG_sigmas = (1.0, 10.0)

    t0 = time.time()

    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang, Nxy, pixel_size)

    # Get patterns
    idx = utilities.get_index(point, size, ang_data)
    pats = utilities.get_patterns(pat_obj, idx)

    # Process patterns
    pats = utilities.process_patterns(pats, equalize=True, dog_sigmas=DoG_sigmas)

    # Get initial guesses
    R = pats[50, 50]
    p0 = get_initial_guess(R, pats)

    # Get homographies
    subset_slice = (slice(28, -28), slice(28, -28))
    p = get_homography(R, pats, subset_slice=subset_slice, p0=p0, max_iter=50, conv_tol=1e-7)
    np.save(f"{save_name}_p.npy", p)

    # Get deformation gradients and strain
    Fe = homography_to_elastic_deformation(p, ang_data.pc)
    e, w = deformation_to_strain(Fe)

    # Print time
    t1 = time.time()
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f">Total time for the entire run: {hours:.0f}:{minutes:.0f}:{seconds:.0f}")

    # Save the results
    xy = (50, 50) # The location of the reference point
    utilities.view_tensor_images(Fe, tensor_type="deformation", xy=xy, save_name=save_name, save_dir="results/")
    utilities.view_tensor_images(e, tensor_type="strain", xy=xy, save_name=save_name, save_dir="results/")
    utilities.view_tensor_images(w, tensor_type="rotation", xy=xy, save_name=save_name, save_dir="results/")