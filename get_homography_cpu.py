import os
from collections import namedtuple
from enum import Enum
from typing import Union, Callable
import contextlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, interpolate, signal
from tqdm.auto import tqdm
import joblib
from joblib import Parallel, delayed
import mpire
import dill

import utilities
import warp
import conversions
import Data


# Type hints
PATS = Union[Data.UP2, np.ndarray]
ARRAY = Union[np.ndarray, list, tuple]
NUMBER = Union[int, float]

# Create some namedtuples
ICGN_Pre = namedtuple("ICGN_Pre", ["r", "r_zmsv", "NablaR_dot_Jac", "c", "L", "xi", "x", "y"])
FMTCC_Pre = namedtuple("FMTCC_Pre", ["r_init", "r_fft", "r_fmt", "x_fmt", "y_fmt", "X_fmt", "Y_fmt"])


class ICGNOptimizer:

    def __init__(
        self,
        pat_obj: namedtuple,
        x0: ARRAY,
        PC: ARRAY,
        sample_tilt: NUMBER,
        detector_tilt: NUMBER,
        pixel_size: NUMBER,
        step_size: NUMBER,
        scan_shape: ARRAY,
        small_strain: bool = False,
        C: np.ndarray = None,
        fixed_projection: bool = True,
        traction_free: bool = True,
    ) -> None:

        # Scan parameters
        self.PC = PC
        self.scan_shape = scan_shape
        self.pidx = np.arange(scan_shape[0] * scan_shape[1]).reshape(scan_shape)
        self.x0 = self.pidx[x0]

        # Optimizer parameters
        self.pat_obj = pat_obj
        self.conv_tol = 1e-3
        self.max_iter = 50

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
        self.verbose = False
        self.extra_verbose = False

        # Run the initialization functions
        self.results = utilities.Results(
            scan_shape, PC, x0,
            step_size / pixel_size,
            fixed_projection,
            detector_tilt,
            sample_tilt,
            traction_free,
            small_strain,
            C,
        )
        self.set_roi()
        self.set_image_processing_kwargs()
        self.set_homography_subset(int(pat_obj.patshape[0] * 0.7), "image")

    def set_image_processing_kwargs(
        self, low_pass_sigma: NUMBER = 2.5, high_pass_sigma: NUMBER = 101.0, truncate_std_scale: NUMBER = 3.0
    ) -> None:
        self.image_processing_kwargs = dict = {
            "high_pass_sigma": high_pass_sigma,
            "low_pass_sigma": low_pass_sigma,
            "truncate_std_scale": truncate_std_scale,
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

    def view_reference(self) -> None:
        R0 = self.pat_obj.read_pattern(self.x0, process=False)
        R1 = self.pat_obj.read_pattern(self.x0, process=True, p_kwargs=self.image_processing_kwargs)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(R0, cmap="gray")
        ax[0].set_title("Raw Reference")
        ax[0].axis("off")
        ax[1].imshow(R1, cmap="gray")
        ax[1].set_title("Processed Reference")
        ax[1].axis("off")
        plt.show()

    def rc2idx(self, row: NUMBER, col: NUMBER) -> NUMBER:
        """Convert x, y coordinates to an index."""
        return row * self.scan_shape[1] + col

    def idx2rc(self, idx: NUMBER) -> tuple:
        """Convert an index to row, column coordinates."""
        row = idx // self.scan_shape[1]
        col = idx % self.scan_shape[1]
        return np.array([row, col])

    def reference_precompute(self) -> None:
        """Precompute arrays/values for the reference subset for the IC-GN algorithm."""
        # Get the reference image and process it
        self.R = self.pat_obj.read_pattern(self.x0, process=True, p_kwargs=self.image_processing_kwargs)

        # Get coordinates
        x = np.arange(self.R.shape[1]) - self.h0[0]
        y = np.arange(self.R.shape[0]) - self.h0[1]
        X, Y = np.meshgrid(x, y, indexing="xy")
        xi = np.array([X[self.subset_slice].flatten(), Y[self.subset_slice].flatten()])
        # xi = np.array([Y[self.subset_slice].flatten(), X[self.subset_slice].flatten()])

        # Compute the intensity gradients of the subset
        spline = interpolate.RectBivariateSpline(x, y, self.R.T, kx=5, ky=5)
        # spline = interpolate.RectBivariateSpline(x, y, self.R, kx=5, ky=5)
        GRx = spline(xi[0], xi[1], dx=1, dy=0, grid=False)
        GRy = spline(xi[0], xi[1], dx=0, dy=1, grid=False)
        GR = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN
        # GR = np.vstack((GRy, GRx)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN
        r = spline(xi[0], xi[1], grid=False).flatten()
        r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
        r = (r - r.mean()) / r_zmsv

        # Compute the jacobian of the shape function
        _1 = np.ones(xi.shape[1])
        _0 = np.zeros(xi.shape[1])
        out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0] ** 2, -xi[1] * xi[0]]])
        out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0] * xi[1], -xi[1] ** 2]])
        Jac = np.vstack((out0, out1))  # 2x8xN

        # Multiply the gradients by the jacobian
        NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 1x8xN -> 8xN
        H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)

        # Compute the Cholesky decomposition
        (c, L) = linalg.cho_factor(H)

        # Store the precomputed values
        del GR, GRx, GRy, Jac, H, X, Y, spline
        self.icgn_pre = ICGN_Pre(r, r_zmsv, NablaR_dot_Jac, c, L, xi, x, y)

        # Precompute the FMT-FCC initial guess
        if self.init_type is not None and self.init_type != "none":
            # Get the FMT-FCC initial guess precomputed items
            r_init = window_and_normalize(self.R[self.guess_subset_slice])
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
            self.fmtcc_pre = FMTCC_Pre(r_init, r_fft, r_FMT, x_fmt, y_fmt, X_fmt, Y_fmt)

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
            with mpire.WorkerPool(n_jobs=n_cores, use_dill=True, start_method="spawn") as pool:
                out = pool.map(self.run_single, [i for i in indices], progress_bar=True, iterable_len=self.num_points)
        else:
            out = [self.run_single(idx) for idx in tqdm(indices)]
        # Store the results
        self.results.update(out, self.roi)

    def run_single(self, idx: int) -> np.ndarray:
        """Run the homography optimization for a single point."""
        # Get the target image and process it
        T = self.pat_obj.read_pattern(idx, process=True, p_kwargs=self.image_processing_kwargs)
        T_spline = interpolate.RectBivariateSpline(
            self.icgn_pre.x, self.icgn_pre.y, T.T, kx=5, ky=5
        #     self.icgn_pre.x, self.icgn_pre.y, T, kx=5, ky=5
        )
        ### T_spline = warp.Spline(T, 5, 5, self.h0, self.subset_slice)
        # Run initial guess
        if self.init_type is None or self.init_type == "none":
            p = np.zeros(8, dtype=float)
        else:
            # Window and normalize the target
            t_init = window_and_normalize(T[self.guess_subset_slice])
            ### T_spline_init = warp.Spline(t_init, 5, 5, self.PC, None)
            # Do the angle search first
            t_init_fft = np.fft.fftshift(np.fft.fft2(t_init))
            t_init_FMT, _ = FMT(
                t_init_fft,
                self.fmtcc_pre.X_fmt,
                self.fmtcc_pre.Y_fmt,
                self.fmtcc_pre.x_fmt,
                self.fmtcc_pre.y_fmt,
            )
            cc = signal.fftconvolve(
                self.fmtcc_pre.r_fmt, t_init_FMT[::-1], mode="same"
            ).real
            theta = (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)
            # Apply the rotation
            h = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
            ### t_init_rot = T_spline_init.warp(h).reshape(t_init.shape)
            t_init_rot = warp.deform_image(t_init, h, self.PC)
            # Do the translation search
            cc = signal.fftconvolve(
                self.fmtcc_pre.r_init, t_init_rot[::-1, ::-1], mode="same"
            ).real
            shift = np.unravel_index(np.argmax(cc), cc.shape) - np.array(cc.shape) / 2
            # Store the homography
            measurement = np.array([[-shift[0], -shift[1], -theta]])
            # Convert the measurements to homographies
            if self.init_type == "full":
                p = conversions.xyt2h(measurement, self.PC)
            else:
                p = conversions.xyt2h_partial(measurement)

        # Run the optimization
        num_iter = 0
        norms = []
        residuals = []
        while num_iter < self.max_iter:
            # Warp the target subset
            num_iter += 1
            t_deformed = warp.deform(self.icgn_pre.xi, T_spline, p)
            t_mean = t_deformed.mean()
            t_deformed = (t_deformed - t_mean) / np.sqrt(((t_deformed - t_mean) ** 2).sum())
            # Compute the residuals
            e = self.icgn_pre.r - t_deformed
            residuals.append(np.abs(e).mean())
            # Copmute the gradient of the correlation criterion
            dC_IC_ZNSSD = (
                2
                / self.icgn_pre.r_zmsv
                * np.matmul(e, self.icgn_pre.NablaR_dot_Jac.T)
            )  # 8x1
            # Find the deformation incriment, delta_p, by solving the linear system
            # H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
            dp = linalg.cho_solve(
                (self.icgn_pre.c, self.icgn_pre.L), -dC_IC_ZNSSD.reshape(-1, 1)
            )[:, 0]
            # Update the parameters
            norm = dp_norm(dp, self.icgn_pre.xi)
            Wp = warp.W(p)
            Wdp = warp.W(dp)
            Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
            p = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]
            # Store the update
            norms.append(norm)
            # print(f"Pattern {idx}: Iteration {num_iter}, Norm: {norm:.4f}, Residual: {residuals[-1]:.4f}")
            if norm < self.conv_tol:
                break
        if self.verbose:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(self.icgn_pre.r.reshape(self.scan_shape), cmap="gray")
            ax[0].set_title("Reference")
            ax[0].axis("off")
            ax[1].imshow(t_deformed.reshape(self.scan_shape), cmap="gray")
            ax[1].set_title("Deformed Target")
            ax[1].axis("off")
            ax[2].imshow(np.abs(e).reshape(self.scan_shape), cmap="gray")
            ax[2].set_title("Residuals")
            ax[2].axis("off")
            plt.tight_layout()
            plt.savefig(f"./gif/{idx}_CPU.png")
            plt.close("all")
        
        if num_iter >= self.max_iter and self.verbose:
            # print(f"Warning: Maximum number of iterations reached for pattern {idx}!")
            row = int(self.icgn_pre.xi[0].max() - self.icgn_pre.xi[0].min() + 1)
            col = int(self.icgn_pre.xi[1].max() - self.icgn_pre.xi[1].min() + 1)
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(self.icgn_pre.r.reshape(row, col), cmap="gray")
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
            if self.init_type is not None and self.init_type != "none":
                h = conversions.xyt2h_partial(measurement)[0]
                tar_rot_shift = warp.deform_image(t_init_rot, h, self.PC)
                fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
                ax[0, 0].imshow(self.fmtcc_pre.r_init, cmap="gray")
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
                ax[1, 1].imshow(self.fmtcc_pre.r_init - t_init, cmap="gray")
                ax[1, 1].set_title("Difference")
                ax[1, 2].imshow(self.fmtcc_pre.r_init - t_init_rot, cmap="gray")
                ax[1, 2].set_title("Difference")
                ax[1, 3].imshow(self.fmtcc_pre.r_init - tar_rot_shift, cmap="gray")
                ax[1, 3].set_title("Difference")
                plt.tight_layout()
                plt.savefig(f"gif/FMT-FCC_{idx}.png")
                plt.close(fig)

        return p, int(num_iter), float(residuals[-1]), float(norms[-1])

    def save(self, filename: str) -> None:
        """Save the optimizer to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def load(self, filename: str) -> None:
        """Load the optimizer from a file."""
        with open(filename, "rb") as f:
            obj = dill.load(f)
        self.__dict__.update(obj.__dict__)


# Make a Enum class for the different types of homography initialization
class InitType(Enum):
    NONE: str = "none"
    FULL: str = "full"
    PARTIAL: str = "partial"


# Context manager to patch joblib to report into tqdm progress bar given as argument
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


### Functions for the inverse composition gauss-newton algorithm

def optimize(
    pats: PATS,
    x0: ARRAY,
    init_type: InitType = InitType.NONE,
    crop_fraction: float = 0.7,
    max_iter: int = 50,
    conv_tol: float = 1e-3,
    n_jobs: int = -1,
    verbose: bool = False,
    ) -> np.ndarray:
    """Routine for running the inverse composition gauss-newton algorithm.
    
    Args:
        pats (Data.UP2 or np.ndarray): The patterns to optimize. If array, last two dimensions should be the pattern. Shape is (..., H, W).
        x0 (array-like): The coordinate of the reference pattern. Can be an integer, list, tuple, or np.ndarray.
        init_type (InitType): The type of initial guess to use. Can be "none", "full", or "partial".
        crop_fraction (float): The fraction of the pattern to use for the homography optimization. Must be between 0 and 1.
        max_iter (int): The maximum number of iterations to run.
        conv_tol (float): The convergence tolerance.
        verbose (bool): Whether to print progress messages.

    Returns:
        homographies (np.ndarray): The optimized homography parameters. Shape matches pattern input.
        iterations (np.ndarray): The number of iterations taken to converge for each pattern. Shape matches pattern input.
        residuals (np.ndarray): The final residuals for each pattern. Shape matches pattern input.
        dp_norms (np.ndarray): The final deformation increment norms for each pattern. Shape matches pattern input."""

    ### Prepare the inputs ###
    # Check the crop fraction
    if crop_fraction <= 0 or crop_fraction >= 1:
        raise ValueError("Crop fraction must be between 0 and 1.")

    # Check the reference coordinate, convert to a 1D idx if needed
    if type(x0) in [list, tuple, np.ndarray] and len(x0) == 1:
        x0 = x0[0]
    elif type(x0) in [list, tuple, np.ndarray] and len(x0) != len(out_shape):
        raise ValueError("The dimension of x0 must match the dimension of pats. Got {len(x0)} for x0 and {len(out_shape)} for dataset shape.")
    elif type(x0) in [list, tuple, np.ndarray] and len(x0) == len(out_shape):
        # Unravel 2D coordinates to 1D indices
        x0 = np.ravel_multi_index(x0, out_shape)

    # Check convergence parameters
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be greater than 0.")
    if conv_tol <= 0:
        raise ValueError("Convergence tolerance must be greater than 0.")

    # Check the init type, must be an instance of the InitType enum or a string matching the values of the enum
    if not isinstance(init_type, InitType):
        if type(init_type) == str:
            init_type = InitType(init_type.lower())
        else:
            raise TypeError("init_type must be an instance of the InitType enum or a string matching the values of the enum.")

    # Check the number of jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() - 1

    # Prepare the patterns
    if type(pats) == Data.UP2:
        N = pats.nPatterns
        out_shape = (pats.nPatterns,)
        patshape = pats.patshape
        get_pat = lambda idx: pats.read_pattern(idx, process=True)
    elif type(pats) == np.ndarray:
        N = np.prod(pats.shape[:-2])
        out_shape = pats.shape[:-2]
        patshape = pats.shape[-2:]
        pats = pats.reshape(-1, pats.shape[-2], pats.shape[-1])
        get_pat = lambda idx: pats[idx]
    else:
        raise TypeError("pats must be a Data.UP2 object or a numpy array.")
    h0 = (patshape[1] // 2, patshape[0] // 2)
    crop_row = int(patshape[0] * (1 - crop_fraction) / 2)
    crop_col = int(patshape[1] * (1 - crop_fraction) / 2)
    subset_slice = (slice(crop_row, -crop_row), slice(crop_col, -crop_col))


    ### Reference precompute ###
    # Get the reference image
    R = get_pat(x0)

    # Get coordinates
    x = np.arange(R.shape[1]) - h0[0]
    y = np.arange(R.shape[0]) - h0[1]
    X, Y = np.meshgrid(x, y, indexing="xy")
    xi = np.array([Y[subset_slice].flatten(), X[subset_slice].flatten()])

    # Compute the intensity gradients of the subset
    ref_spline = interpolate.RectBivariateSpline(x, y, R, kx=5, ky=5)
    GRx = ref_spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    GRy = ref_spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    GR = np.vstack((GRy, GRx)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN
    r = ref_spline(xi[0], xi[1], grid=False).flatten()
    r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
    r = (r - r.mean()) / r_zmsv

    # Compute the jacobian of the shape function
    _1 = np.ones(xi.shape[1])
    _0 = np.zeros(xi.shape[1])
    out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0] ** 2, -xi[1] * xi[0]]])
    out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0] * xi[1], -xi[1] ** 2]])
    Jac = np.vstack((out0, out1))  # 2x8xN

    # Multiply the gradients by the jacobian
    NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 1x8xN -> 8xN
    H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)

    # Compute the Cholesky decomposition
    cho_params = linalg.cho_factor(H)

    # Store the precomputed values
    del GR, GRx, GRy, Jac, H, X, Y, ref_spline


    #### Precompute the FMT-FCC initial guess ###
    if init_type is not InitType.NONE:
        _s  = 2**(min(patshape).bit_length() - 1)
        row_start = (patshape[0] - _s) // 2
        col_start = (patshape[1] - _s) // 2
        init_guess_subset_slice = (slice(row_start, row_start + _s), slice(col_start, col_start + _s))
        # Get the FMT-FCC initial guess precomputed items
        r_init = window_and_normalize(R[init_guess_subset_slice])
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
        r_fmt, _ = FMT(r_fft, X_fmt, Y_fmt, x_fmt, y_fmt)

    ### Run the optimization in parallel ###
    if verbose:
        with tqdm_joblib(tqdm(total=N, desc="Patterns optimized")) as progress_bar:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_process_single_pattern)(
                    idx, get_pat, init_type, 
                    r_init if init_type is not InitType.NONE else None,
                    r_fmt if init_type is not InitType.NONE else None,
                    X_fmt if init_type is not InitType.NONE else None,
                    Y_fmt if init_type is not InitType.NONE else None,
                    x_fmt if init_type is not InitType.NONE else None,
                    y_fmt if init_type is not InitType.NONE else None,
                    r, r_zmsv, xi, NablaR_dot_Jac, cho_params, h0, max_iter, conv_tol
                ) for idx in range(N)
            )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_pattern)(
                idx, get_pat, init_type, 
                r_init if init_type is not InitType.NONE else None,
                r_fmt if init_type is not InitType.NONE else None,
                X_fmt if init_type is not InitType.NONE else None,
                Y_fmt if init_type is not InitType.NONE else None,
                x_fmt if init_type is not InitType.NONE else None,
                y_fmt if init_type is not InitType.NONE else None,
                r, r_zmsv, xi, NablaR_dot_Jac, cho_params, h0, max_iter, conv_tol
            ) for idx in range(N)
        )

    # Unpack results
    homographies = np.zeros((N, 8), dtype=float)
    homographies_guess = np.zeros((N, 8), dtype=float)
    iterations = np.zeros(N, dtype=int)
    residuals = np.zeros(N, dtype=float)
    dp_norms = np.zeros(N, dtype=float)
    
    for idx, (h, p_guess, num_iter, residual, dp_norm) in enumerate(results):
        homographies[idx] = h
        homographies_guess[idx] = p_guess
        iterations[idx] = num_iter
        residuals[idx] = float(residual)
        dp_norms[idx] = float(dp_norm)

    # Reshape the results to match the input pattern shape
    homographies = homographies.reshape(out_shape + (8,))
    homographies_guess = homographies_guess.reshape(out_shape + (8,))
    iterations = iterations.reshape(out_shape)
    residuals = residuals.reshape(out_shape)
    dp_norms = dp_norms.reshape(out_shape)
    
    # Return the results
    if init_type is not InitType.NONE:
        return homographies, homographies_guess, iterations, residuals, dp_norms
    else:
        return homographies, iterations, residuals, dp_norms


def _process_single_pattern(idx, get_pat, init_type, r_init, r_fmt, X_fmt, Y_fmt, x_fmt, y_fmt, 
                          r, r_zmsv, xi, NablaR_dot_Jac, cho_params, h0, max_iter, conv_tol):
    """Helper function to process a single pattern for parallel execution."""
    # Run initial guess
    if init_type == InitType.NONE:
        h = np.zeros(8, dtype=float)
    else:
        measurement = initial_guess_run(get_pat, idx, r_init, r_fmt, X_fmt, Y_fmt, x_fmt, y_fmt)
        if init_type == InitType.FULL:
            h = conversions.xyt2h(measurement, h0)
        else:
            h = conversions.xyt2h_partial(measurement)
    
    initial_guess = h.copy()

    # Run the optimization
    h, num_iter, residual, dp_norm = optimize_run(
        get_pat, idx, h, r, r_zmsv, xi, NablaR_dot_Jac, cho_params, 
        max_iter=max_iter, conv_tol=conv_tol, return_full=False
    )
    
    return h, initial_guess, num_iter, residual, dp_norm


def optimize_run(
    get_pat: Callable,
    idx: int,
    h: np.ndarray,
    r: np.ndarray,
    r_zmsv: float,
    xi: np.ndarray,
    NablaR_dot_Jac: np.ndarray,
    cho_params: tuple,
    max_iter: int = 50,
    conv_tol: float = 1e-3,
    return_full: bool = False,
) -> tuple:
    """Run the homography optimization for a single point.
    
    Args:
        get_pat (Callable): Function to get the target image.
        idx (int): Index of the target image.
        h (np.ndarray): The initial guess of the homography.
        r (np.ndarray): The reference subset.
        r_zmsv (float): The zero mean, unit variance normalization factor for the reference subset.
        xi (np.ndarray): The coordinates of the reference subset.
        NablaR_dot_Jac (np.ndarray): The gradient of the reference subset.
        cho_params (tuple): The Cholesky decomposition parameters.
        max_iter (int): The maximum number of iterations to run.
        conv_tol (float): The convergence tolerance.
        return_full (bool): Whether to return the full optimization results.
    Returns:
        p (np.ndarray): The optimized homography parameters.
        num_iter (int): The number of iterations taken to converge.
        residuals (float or list): The final residuals for each iteration. If return_full is False, returns the final residual.
        norms (float or list): The final deformation increment norms for each iteration. If return_full is False, returns the final norm."""
    # Get the target image
    T = get_pat(idx)
    h0 = (T.shape[1] // 2, T.shape[0] // 2)
    x = np.arange(T.shape[1]) - h0[0]
    y = np.arange(T.shape[0]) - h0[1]
    T_spline = interpolate.RectBivariateSpline(
        x, y, T, kx=5, ky=5
    )

    # Run the optimization
    num_iter = 0
    norms = []
    residuals = []
    while num_iter < max_iter:
        # Warp the target subset
        num_iter += 1
        t_deformed = warp.deform(xi, T_spline, h)
        t_mean = t_deformed.mean()
        t_deformed = (t_deformed - t_mean) / np.sqrt(((t_deformed - t_mean) ** 2).sum())
        # Compute the residuals
        e = r - t_deformed
        residuals.append(np.abs(e).mean())
        # Copmute the gradient of the correlation criterion
        dC_IC_ZNSSD = (
            2
            / r_zmsv
            * np.matmul(e, NablaR_dot_Jac.T)
        )  # 8x1
        # Find the deformation incriment, delta_p, by solving the linear system
        # H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
        dp = linalg.cho_solve(
            cho_params, -dC_IC_ZNSSD.reshape(-1, 1)
        )[:, 0]
        # Update the parameters
        norm = dp_norm(dp, xi)
        Wp = warp.W(h)
        Wdp = warp.W(dp)
        Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
        h = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]
        # Store the update
        norms.append(norm)
        if norm < conv_tol:
            break

    if return_full:
        return h, num_iter, residuals, norms
    else:
        return h, num_iter, residuals[-1], norms[-1]
    

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


### Functions for the global cross-correlation initial guess

def initial_guess_run(
    get_pat: Callable,
    idx: int,
    r_init: np.ndarray,
    r_fmt: np.ndarray,
    X_fmt: np.ndarray,
    Y_fmt: np.ndarray,
    x_fmt: np.ndarray,
    y_fmt: np.ndarray,
) -> np.ndarray:
    """Run the initial guess optimization for a single point.
    
    Args:
        get_pat (Callable): Function to get the target image.
        idx (int): Index of the target image.
        r_init (np.ndarray): The reference subset.
        r_FMT (np.ndarray): The Fourier-Mellin Transform of the reference subset.
        X_fmt (np.ndarray): The x-coordinates of the reference subset in log-polar coordinates.
        Y_fmt (np.ndarray): The y-coordinates of the reference subset in log-polar coordinates.
        x_fmt (np.ndarray): The x-coordinates of the output image in log-polar coordinates.
        y_fmt (np.ndarray): The y-coordinates of the output image in log-polar coordinates.

    Returns:
        measurement (np.ndarray): An initial guess of the shift and rotation of the target image. [shift[0], shift[1], theta]"""
    # Get the target image
    T = get_pat(idx)
    h0 = (T.shape[1] // 2, T.shape[0] // 2)
    t_init = window_and_normalize(T)
    # Do the angle search first
    t_init_fft = np.fft.fftshift(np.fft.fft2(t_init))
    t_init_FMT, _ = FMT(t_init_fft, X_fmt, Y_fmt, x_fmt, y_fmt)
    cc = signal.fftconvolve(r_fmt, t_init_FMT[::-1], mode="same").real
    theta = (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)
    # Apply the rotation
    h = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
    t_init_rot = warp.deform_image(t_init, h, h0)
    # Do the translation search
    cc = signal.fftconvolve(r_init, t_init_rot[::-1, ::-1], mode="same").real
    shift = np.unravel_index(np.argmax(cc), cc.shape) - np.array(cc.shape) / 2
    # Store the homography
    measurement = np.array([[-shift[0], -shift[1], -theta]])
    
    return measurement


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
