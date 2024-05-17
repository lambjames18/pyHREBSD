import numpy as np
from scipy import interpolate, signal, ndimage
from tqdm.auto import tqdm


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


def GCC_Initial_Guess(ref: np.ndarray, targets: np.ndarray):
    """Perform a global cross-correlation initial guess for the homographies of the targets.
    Rotation is determined using a Fourier-Mellin Transform.
    Translation is determined using a 2D cross-correlation.
    The images are cropped to 128x128 to speed up the process.
    Args:
        ref (np.ndarray): The reference pattern. (H, W)
        targets (np.ndarray): The target patterns. (M, N, H, W) or (N, H, W) or (H, W) for example.
    Returns:
        np.ndarray: The homographies of the targets. (M, N, 8) or (N, 8) or (8) for example."""
    # Check inputs
    targets = np.array(targets)
    if targets.ndim == 4:
        full_shape = targets.shape
        targets = targets.reshape(full_shape[0] * full_shape[1], full_shape[2], full_shape[3])
    elif targets.ndim == 2:
        targets = targets.reshape(1, targets.shape[0], targets.shape[1])
        full_shape = None
    else:
        full_shape = None

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
    for i in tqdm(range(len(targets))):
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

    # Reshape the angles
    if full_shape is not None:
        homographies = homographies.reshape(full_shape[:2] + (8,))
    else:
        homographies = np.squeeze(homographies)
    return homographies