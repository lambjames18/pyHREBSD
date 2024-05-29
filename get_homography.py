from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, interpolate, signal
from tqdm.auto import tqdm
import mpire
import dill

import utilities
import warp
import conversions

# Type hints
ARRAY = np.ndarray | list | tuple
NUMBER = int | float

# Get the number of physical cores
NUM_PHYSICAL_CORES = int(mpire.cpu_count() // 2)

# Create some namedtuples
ReferencePrecompute = namedtuple(
    "ReferencePrecompute",
    [
        "r",
        "r_zmsv",
        "NablaR_dot_Jac",
        "c",
        "L",
        "xi",
        "x",
        "y",
        "r_init",
        "r_fft",
        "r_fmt",
        "x_fmt",
        "y_fmt",
        "X_fmt",
        "Y_fmt",
    ],
)
ICGNResults = namedtuple(
    "ICGNResults",
    ["num_iter", "residuals", "norms", "homographies", "F", "e", "w", "s"],
)


class ICGNOptimizer:

    def __init__(
        self,
        pat_obj: namedtuple,
        x0: ARRAY,
        PC: ARRAY,
        sample_tilt: NUMBER,
        detector_tilt: NUMBER,
        scan_shape: ARRAY,
        small_strain: bool = False,
        C: np.ndarray = None,
    ) -> None:
        # Get inputs
        self.pat_obj = pat_obj
        self.PC = PC
        self.tilt = 90 - sample_tilt + detector_tilt
        self.scan_shape = scan_shape
        self.conv_tol = 1e-3
        self.max_iter = 50
        self.pidx = np.arange(scan_shape[0] * scan_shape[1]).reshape(scan_shape)
        self.x0 = self.pidx[x0]
        self.C = C
        self.small_strain = small_strain

        # Set empty attributes
        self.results = None
        self.R = None
        self.precomputed_data = None
        self.h0 = None
        self.subset_size = None
        self.subset_slice = None
        self.roi = None
        self.num_points = None
        self.image_processing_kwargs = None
        self.homographies = None
        self.C = None
        self.verbose = False

        # Run the initialization functions
        self.set_roi()
        self.set_image_processing_kwargs()
        self.set_homography_subset(int(pat_obj.patshape[0] * 0.7), "image")

    def set_image_processing_kwargs(
        self, sigma: NUMBER = None, equalize: bool = True, truncate: bool = True
    ) -> None:
        if sigma is None:
            sigma = self.pat_obj.patshape[0] // 100
        self.image_processing_kwargs = dict = {
            "sigma": sigma,
            "equalize": equalize,
            "truncate": truncate,
        }

    def set_roi(self, start: ARRAY = None, span: ARRAY = None, mask: np.ndarray = None):
        """Set the region of interest for the homography optimization."""
        if mask is not None:
            self.roi = mask
        else:
            if start is None:
                start = (0, 0)
            if span is None:
                span = self.scan_shape
            self.roi = np.zeros(self.scan_shape, dtype=bool)
            self.roi[start[0] : start[0] + span[0], start[1] : start[1] + span[1]] = (
                True
            )
        self.num_points = self.roi.sum()

    def get_roi_bbox(self) -> tuple:
        """Get the bounding box of the region of interest."""
        r0, c0 = np.array(np.where(self.roi)).min(axis=1)
        r1, c1 = np.array(np.where(self.roi)).max(axis=1)
        return (r0, r1, c0, c1)

    def set_homography_subset(self, size: NUMBER, center: tuple | str) -> None:
        """Set the homography center and the size of the subset"""
        if type(center) == str and center.lower() == "image":
            self.h0 = (self.pat_obj.patshape[1] / 2, self.pat_obj.patshape[0] / 2)
        elif type(center) == str and center.lower() == "pattern":
            self.h0 = self.PC
        else:
            self.h0 = center
        self.subset_size = size
        r0 = int(max(0, self.h0[1] - size // 2))
        r1 = int(min(self.pat_obj.patshape[0], self.h0[1] + size // 2))
        c0 = int(max(0, self.h0[0] - size // 2))
        c1 = int(min(self.pat_obj.patshape[1], self.h0[0] + size // 2))
        self.subset_slice = (slice(r0, r1), slice(c0, c1))

    def set_initial_guess_params(
        self, subset_size: NUMBER = None, init_type: str = "full"
    ) -> None:
        """Set the initial guess parameters for the homography optimization.
        The subset size must be a power of 2."""
        if init_type.lower() not in ["full", "partial", "none"]:
            raise ValueError(
                "Invalid initial guess type. Must be 'full', 'partial', or 'none'."
            )
        else:
            self.init_type = init_type
        if subset_size is None:
            subset_size = int(self.pat_obj.patshape[0].bit_length() - 1)
        else:
            if subset_size > self.pat_obj.patshape[0]:
                raise ValueError("Subset size cannot be larger than the pattern size.")
            elif subset_size < 128:
                raise ValueError("Subset size must be at least 128.")
            elif (subset_size != 0) and (subset_size & (subset_size - 1) == 0):
                pass
            else:
                raise ValueError("Subset size must be a power of 2.")
        c = np.array(self.pat_obj.patshape) // 2
        self.guess_subset_slice = (
            slice(c[0] - subset_size // 2, c[0] + subset_size // 2),
            slice(c[1] - subset_size // 2, c[1] + subset_size // 2),
        )

    def print_setup(self) -> None:
        """Print the setup of the homography optimization."""
        r0, r1, c0, c1 = self.get_roi_bbox()
        x0_rc = self.idx2rc(self.x0)
        print("***********************************")
        print("Dataset information:")
        print(f"\tPattern size: {self.pat_obj.patshape} pixels")
        print(f"\tScan shape: {self.scan_shape} pixels")
        print(f"\tPC: {self.PC}")
        print(f"\tSample-to-detector tilt: {self.tilt} degrees")
        print("Optimization parameters:")
        print(f"\tROI: Rows {r0} to {r1}, Columns {c0} to {c1}")
        print(f"\tROI: {self.pidx[r0, c0]} to {self.pidx[r1, c1]} (indices)")
        print(f"\tReference location: {x0_rc} (row/column)")
        print(f"\tReference location: {self.x0} (index)")
        print(f"\tSubset size: {self.subset_size} pixels")
        print(f"\tHomography center: {self.h0}")
        print(f"\tROI shape: {self.roi.shape}")
        print(f"\tNumber of points: {self.num_points}")
        print(f"\tInitial guess type: {self.init_type}")
        print(f"\tMaximum iterations: {self.max_iter}")
        print(f"\tConvergence tolerance: {self.conv_tol}")
        print("Image processing parameters:")
        print(f"\tImage processing kwargs: {self.image_processing_kwargs}")
        print(f"\tSmall strain: {self.small_strain}")
        print("***********************************")

    def rc2idx(self, row: NUMBER, col: NUMBER) -> NUMBER:
        """Convert x, y coordinates to an index."""
        return row * self.scan_shape[1] + col

    def idx2rc(self, idx: NUMBER) -> tuple:
        """Convert an index to row, column coordinates."""
        row = idx // self.scan_shape[1]
        col = idx % self.scan_shape[1]
        return row, col

    def reference_precompute(self) -> None:
        """Precompute arrays/values for the reference subset for the IC-GN algorithm."""
        # Get the reference image and process it
        self.R = utilities.get_pattern(self.pat_obj, self.x0)
        self.R = utilities.process_pattern(self.R, **self.image_processing_kwargs)
        # Get the IC-GN sub-pixel alignment precomuted items
        # Get coordinates
        x = np.arange(self.R.shape[1]) - self.h0[0]
        y = np.arange(self.R.shape[0]) - self.h0[1]
        X, Y = np.meshgrid(x, y)
        xi = np.array([Y[self.subset_slice].flatten(), X[self.subset_slice].flatten()])
        # Compute the intensity gradients of the subset
        spline = warp.Spline(self.R, 5, 5, self.h0, self.subset_slice)
        GR = spline.gradient()
        r = spline(xi[0], xi[1], grid=False, normalize=False).flatten()
        # spline = interpolate.RectBivariateSpline(x, y, self.R, kx=5, ky=5)
        # GRx = spline(xi[0], xi[1], dx=1, dy=0, grid=False)
        # GRy = spline(xi[0], xi[1], dx=0, dy=1, grid=False)
        # GR = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN
        # r = spline(xi[0], xi[1], grid=False).flatten()
        r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
        r = (r - r.mean()) / r_zmsv
        # Compute the jacobian of the shape function
        _1 = np.ones(xi.shape[1])
        _0 = np.zeros(xi.shape[1])
        out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0] ** 2, -xi[1] * xi[0]]])
        out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0] * xi[1], -xi[1] ** 2]])
        Jac = np.vstack((out0, out1))  # 2x8xN
        # Multiply the gradients by the jacobian
        NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 1x8xN
        # Compute the Hessian
        H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)
        (c, L) = linalg.cho_factor(H)
        # Get the FMT-FCC initial guess precomputed items
        r_init = window_and_normalize(self.R[self.subset_slice])
        # Get the dimensions of the image
        height, width = r_init.shape
        # Create a mesh grid of log-polar coordinates
        theta = np.linspace(0, np.pi, int(height), endpoint=False)
        radius = np.linspace(0, height / 2, int(height + 1), endpoint=False)[1:]
        radius_grid, theta_grid = np.meshgrid(radius, theta, indexing="xy")
        radius_grid = radius_grid.flatten()
        theta_grid = theta_grid.flatten()
        # Convert log-polar coordinates to Cartesian coordinates
        x_fmt = 2 ** (np.log2(height) - 1) + radius_grid * np.cos(theta_grid)
        y_fmt = 2 ** (np.log2(height) - 1) - radius_grid * np.sin(theta_grid)
        # Create a mesh grid of Cartesian coordinates
        X_fmt = np.arange(width)
        Y_fmt = np.arange(height)
        # FFT the reference and get the signal
        r_fft = np.fft.fftshift(np.fft.fft2(r_init))
        r_FMT, _ = FMT(r_fft, X_fmt, Y_fmt, x_fmt, y_fmt)

        # Store the precomputed data
        self.precomputed = ReferencePrecompute(
            r,
            r_zmsv,
            NablaR_dot_Jac,
            c,
            L,
            xi,
            x,
            y,
            r_init,
            r_fft,
            r_FMT,
            x_fmt,
            y_fmt,
            X_fmt,
            Y_fmt,
        )

    def run(
        self, n_cores=NUM_PHYSICAL_CORES, max_iter=50, conv_tol=1e-3, verbose=False
    ) -> np.ndarray:
        """Run the homography optimization."""
        # Precompute the reference subset
        indices = self.pidx[self.roi]
        self.reference_precompute()
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.verbose = verbose
        # Run the optimization
        print("Running the homography optimization...")
        if n_cores > 1:
            with mpire.WorkerPool(n_jobs=n_cores, use_dill=True) as pool:
                out = pool.map(self.run_single, [i for i in indices], progress_bar=True, iterable_len=self.num_points)
        else:
            out = [self.run_single(idx) for idx in tqdm(indices)]
        # Store the results
        self.process_results(out)

    def process_results(self, out) -> None:
        p = np.zeros(self.scan_shape + (8,), dtype=float)
        num_iter = np.zeros(self.scan_shape, dtype=int)
        residuals = np.zeros(self.scan_shape, dtype=float)
        norms = np.zeros(self.scan_shape, dtype=float)
        p[self.roi] = np.array([o[0] for o in out])
        num_iter[self.roi] = np.array([o[1] for o in out])
        residuals[self.roi] = np.array([o[2] for o in out])
        norms[self.roi] = np.array([o[3] for o in out])
        # Get the deformation gradient and the strain
        PC_temp = (self.PC[0] - self.pat_obj.patshape[0] / 2, self.PC[1] - self.pat_obj.patshape[1] / 2, self.PC[2])
        F = np.zeros((self.scan_shape[0], self.scan_shape[1], 3, 3), dtype=float)
        F[self.roi] = conversions.h2F(p[self.roi], PC_temp)
        e = np.zeros((self.scan_shape[0], self.scan_shape[1], 3, 3), dtype=float)
        w = np.zeros((self.scan_shape[0], self.scan_shape[1], 3, 3), dtype=float)
        if self.C is not None:
            E, R, S = conversions.F2strain(F[self.roi], self.C, self.small_strain)
            s = np.zeros((self.scan_shape[0], self.scan_shape[1], 3, 3), dtype=float)
            s[self.roi] = S
        else:
            E, R = conversions.F2strain(F[self.roi], None, self.small_strain)
            s = None
        e[self.roi] = E
        w[self.roi] = R
        self.results = ICGNResults(num_iter, residuals, norms, p, F, e, w, s)      

    def run_single(self, idx: int) -> np.ndarray:
        """Run the homography optimization for a single point."""
        # Get the target image and process it
        T = utilities.get_pattern(self.pat_obj, idx)
        T = utilities.process_pattern(T, **self.image_processing_kwargs)
        # T_spline = interpolate.RectBivariateSpline(
        #     self.precomputed.x, self.precomputed.y, normalize(T), kx=5, ky=5
        # )
        T_spline = warp.Spline(T, 5, 5, self.h0, self.subset_slice)
        # Run initial guess
        # Window and normalize the target
        t_init = window_and_normalize(T[self.subset_slice])
        T_spline_init = warp.Spline(t_init, 5, 5, self.PC, self.guess_subset_slice)
        # Do the angle search first
        t_init_fft = np.fft.fftshift(np.fft.fft2(t_init))
        t_init_FMT, _ = FMT(
            t_init_fft,
            self.precomputed.X_fmt,
            self.precomputed.Y_fmt,
            self.precomputed.x_fmt,
            self.precomputed.y_fmt,
        )
        cc = signal.fftconvolve(
            self.precomputed.r_fmt, t_init_FMT[::-1], mode="same"
        ).real
        theta = (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)
        # Apply the rotation
        h = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
        t_init_rot = T_spline_init.warp(h).reshape(t_init.shape)
        # t_init_rot = warp.deform_image(t_init, h, self.PC)
        # Do the translation search
        cc = signal.fftconvolve(
            self.precomputed.r_init, t_init_rot[::-1, ::-1], mode="same"
        ).real
        shift = np.unravel_index(np.argmax(cc), cc.shape) - np.array(cc.shape) / 2
        # Store the homography
        measurement = np.array([[shift[1], shift[0], -theta]])
        # Convert the measurements to homographies
        if self.init_type == "full":
            p = conversions.xyt2h(measurement, self.PC, self.tilt)
        else:
            p = conversions.xyt2h_partial(measurement)

        # Run the optimization
        num_iter = 0
        norms = []
        residuals = []
        while num_iter < self.max_iter:
            # Warp the target subset
            num_iter += 1
            # t_deformed = warp.deform(self.precomputed.xi, T_spline, p)
            t_deformed = T_spline.warp(p)
            # Compute the residuals
            # e = self.precomputed.r - normalize(t_deformed)
            e = self.precomputed.r - t_deformed
            residuals.append(np.abs(e).mean())
            # Copmute the gradient of the correlation criterion
            dC_IC_ZNSSD = (
                2
                / self.precomputed.r_zmsv
                * np.matmul(e, self.precomputed.NablaR_dot_Jac.T)
            )  # 8x1
            # Find the deformation incriment, delta_p, by solving the linear system
            # H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
            dp = linalg.cho_solve(
                (self.precomputed.c, self.precomputed.L), -dC_IC_ZNSSD.reshape(-1, 1)
            )[:, 0]
            # Update the parameters
            norm = dp_norm(dp, self.precomputed.xi)
            Wp = warp.W(p)
            Wdp = warp.W(dp)
            Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
            p = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]
            # Store the update
            norms.append(norm)
            if norm < self.conv_tol:
                break
        if num_iter >= self.max_iter:
            print(f"Warning: Maximum number of iterations reached for pattern {idx}!")
            row = int(self.precomputed.xi[0].max() - self.precomputed.xi[0].min() + 1)
            col = int(self.precomputed.xi[1].max() - self.precomputed.xi[1].min() + 1)
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(self.precomputed.r.reshape(row, col), cmap="gray")
            ax[0].set_title("Reference")
            ax[1].imshow(t_deformed.reshape(row, col), cmap="gray")
            ax[1].set_title("Deformed target")
            ax[2].imshow(e.reshape(row, col), cmap="gray")
            ax[2].set_title("Final residual")
            for a in ax.flatten():
                a.axis("off")
            plt.tight_layout()
            plt.savefig(f"gif/IC-GN_{idx}.png")
            plt.close(fig)

            h = conversions.xyt2h_partial(measurement)[0]
            tar_rot_shift = warp.deform_image(t_init_rot, h, self.PC)
            fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
            ax[0, 0].imshow(self.precomputed.r_init, cmap="gray")
            ax[0, 0].set_title("Reference")
            ax[1, 0].imshow(cc, cmap="gray")
            ax[1, 0].scatter(cc.shape[1] / 2, cc.shape[0] / 2, c="r", s=100, marker="x")
            ax[1, 0].scatter(
                cc.shape[1] / 2 + shift[1],
                cc.shape[0] / 2 + shift[0],
                c="r",
                s=100,
                marker="*",
            )
            ax[1, 0].set_title("Cross-Correlation")
            ax[0, 1].imshow(t_init, cmap="gray")
            ax[0, 1].set_title("Target")
            ax[0, 2].imshow(t_init_rot, cmap="gray")
            ax[0, 2].set_title("Rotated Target")
            ax[0, 3].imshow(tar_rot_shift, cmap="gray")
            ax[0, 3].set_title("Shifted Rotated Target")
            ax[1, 1].imshow(self.precomputed.r_init - t_init, cmap="gray")
            ax[1, 1].set_title("Difference")
            ax[1, 2].imshow(self.precomputed.r_init - t_init_rot, cmap="gray")
            ax[1, 2].set_title("Difference")
            ax[1, 3].imshow(self.precomputed.r_init - tar_rot_shift, cmap="gray")
            ax[1, 3].set_title("Difference")
            plt.tight_layout()
            plt.savefig(f"gif/FMT-FCC_{idx}.png")
            plt.close(fig)

        return p, int(num_iter), float(residuals[-1]), float(norms[-1])

    def save_results(self, filename: str) -> None:
        """Save the results to a file."""
        with open(filename, "wb") as f:
            dill.dump(self.results, f)

    def save(self, filename: str) -> None:
        """Save the optimizer to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def load_results(self, filename: str) -> None:
        """Load the results from a file."""
        with open(filename, "rb") as f:
            self.results = dill.load(f)

    def load(self, filename: str) -> None:
        """Load the optimizer from a file."""
        with open(filename, "rb") as f:
            obj = dill.load(f)
        self.__dict__.update(obj.__dict__)

# Functions for the inverse composition gauss-newton algorithm


def normalize(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    # img = (img - img.min()) / (img.max() - img.min())
    img_bar = img.mean()
    dimg_tilde = np.sqrt(((img - img_bar) ** 2).sum())
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
    out = np.sqrt(
        (dp_i0**2).sum()
        + (dp_i1**2).sum()
        + (dp_i2**2).sum()
        + (dp[2] ** 2 + dp[5] ** 2)
    )
    return out


# Functions for the global cross-correlation initial guess


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
    variance = (new_normalized_windowed**2).sum(axis=axis) / (
        np.prod(images.shape[-2:]) - 1
    )
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
        np.ndarray: The signal of the Fourier-Mellin Transform. (1D array of length 2**n)
    """
    spline = interpolate.RectBivariateSpline(X, Y, image.real, kx=2, ky=2)
    image_polar = np.abs(spline(x, y, grid=False).reshape(image.shape))
    sig = window_and_normalize(image_polar.mean(axis=1))
    return sig, image_polar
