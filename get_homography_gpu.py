from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, interpolate, signal
from tqdm.auto import tqdm
import mpire
import dill
import torch

import rotations
import utilities
import warp
import conversions
import bspline_gpu as gpu_warp

# Type hints
ARRAY = np.ndarray | list | tuple
NUMBER = int | float

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("**Using GPU**")
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")
    print("**Using MPS**")
else:
    torch.set_default_device("cpu")
    print("**Using CPU**")

# Create some namedtuples
ICGN_Pre = namedtuple("ICGN_Pre", ["r", "r_zmsv", "NablaR_dot_Jac", "L", "xi", "x", "y"])
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
        # Set GPU
        self.device = torch.get_default_device()

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
        self.results = Results(
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
        R = self.pat_obj.read_pattern(self.x0, process=True, p_kwargs=self.image_processing_kwargs)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(R, cmap="gray")
        ax.axis("off")
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
        shape = self.R[self.subset_slice].shape
        R = torch.tensor(self.R, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)

        # Get coordinates
        x = torch.arange(R.shape[3]) - self.h0[0]
        y = torch.arange(R.shape[2]) - self.h0[1]
        X, Y = torch.meshgrid(x, y, indexing="xy")
        xi = torch.stack([Y[self.subset_slice], X[self.subset_slice]], dim=-1).float().unsqueeze(0)  # 1xHxWx2

        # Compute the intensity gradients of the subset and the subset intensities
        bound_fn = gpu_warp.make_bound([0, 0])
        spline_fn = gpu_warp.make_spline([5, 5])
        GR = gpu_warp.grad(R, xi, bound_fn, spline_fn, extrapolate=1)  # 1xHxWx2
        GR = torch.transpose(GR.reshape(1, -1, 2), 1, 2)  # 1x2xN
        r = gpu_warp.pull(R, xi, bound_fn, spline_fn, extrapolate=1)
        r_zmsv = torch.sqrt(((r - r.mean()) ** 2).sum())
        r = ((r - r.mean()) / r_zmsv).reshape(1, 1, *shape)

        # Compute the jacobian of the shape function
        xi0 = xi[0, ..., 0].reshape(-1)
        xi1 = xi[0, ..., 1].reshape(-1)
        _0 = torch.zeros_like(xi1)
        _1 = torch.ones_like(xi1)
        out0 = torch.stack([xi0, xi1, _1, _0, _0, _0, -xi0 ** 2, -xi1 * xi0], dim=0)  # 8xH*W
        out1 = torch.stack([_0, _0, _0, xi0, xi1, _1, -xi0 * xi1, -xi1 ** 2], dim=0)
        Jac = torch.stack([out0, out1], dim=0)  # 2x8xH*W

        # Multiply the gradients by the jacobian
        NablaR_dot_Jac = torch.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 8xH*W
        H = 2 / r_zmsv**2 * torch.matmul(NablaR_dot_Jac, NablaR_dot_Jac.T)  # 8x8

        # Compute the Cholesky decomposition
        # L = torch.linalg.cholesky(H)

        # Store the precomputed values
        del GR, Jac, bound_fn, spline_fn, X, Y
        # return r, r_zmsv, NablaR_dot_Jac, L, xi
        return r, r_zmsv, NablaR_dot_Jac, H, xi
        self.icgn_pre = ICGN_Pre(r, r_zmsv, NablaR_dot_Jac, L, xi, x, y)

        # Precompute the FMT-FCC initial guess
        if self.init_type is not None and self.init_type != "none":
            print("Precomputing the FMT-FCC initial guess...")
            raise NotImplementedError("FMT-FCC initial guess is not implemented yet on the GPU!")

    def run(self, batch_size=32, max_iter=50, conv_tol=1e-3, verbose=False) -> np.ndarray:
        """Run the homography optimization."""
        # Precompute the reference subset
        r, r_zmsv, NablaR_dot_Jac, H, xi = self.reference_precompute()
        # r, r_zmsv, NablaR_dot_Jac, L, xi = self.reference_precompute()
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.verbose = verbose
        # Split indices into batches
        indices = torch.tensor(self.pidx[self.roi].flatten(), dtype=torch.int64).to(self.device)
        indices_split = torch.split(indices, batch_size)
        # Create spline stuff
        bound_fn = gpu_warp.make_bound([0, 0])
        spline_fn = gpu_warp.make_spline([5, 5])
        # Run the optimization
        print("Running the homography optimization...")
        residuals_out = []
        norms_out = []
        num_iters_out = []
        homographies_out = []
        for idx in tqdm(indices_split, desc="ICGN", unit="batch"):
            # Get the target images and process them
            t = self.pat_obj.read_patterns(idx.cpu().numpy(), process=True, p_kwargs=self.image_processing_kwargs)
            t = torch.tensor(t, dtype=torch.float32).to(self.device).unsqueeze(1)
            # Run homography initialization
            if self.init_type is None or self.init_type == "none":
                p = torch.zeros((len(idx), 8), dtype=torch.float32).to(self.device)
            else:
                raise NotImplementedError("Homography initialization on GPU is not implemented yet!")
            # Run the optimization
            num_iter = 0
            while num_iter < max_iter:
                # Warp the target subset
                num_iter += 1
                xi_prime = warp.get_xi_prime_vectorized_gpu(xi[0], p)  # BxHxWx2
                t_deformed = gpu_warp.pull(t, xi_prime, bound_fn, spline_fn, extrapolate=1)
                # Normalize the target
                t_mean = t_deformed.mean(dim=(2, 3), keepdim=True)
                t_deformed = (t_deformed - t_mean) / torch.sqrt(((t_deformed - t_mean) ** 2).sum(dim=(2, 3), keepdim=True))
                # Compute the residuals
                e = (r - t_deformed).reshape(len(idx), -1)  # BxH*W
                residuals = torch.abs(e).mean(dim=1)  # Take residuals over the C, H, W dimensions, leave the batch dimension
                # Compute the gradient of the correlation criterion
                dC_IC_ZNSSD = 2 / r_zmsv * torch.matmul(e, NablaR_dot_Jac.T).reshape(len(idx), 8, 1)  # Bx8
                # Solve H.dot(delta_p) = -dC_IC_ZNSSD using the Cholesky decomposition
                dp = (H.inverse() @ -dC_IC_ZNSSD)[..., 0]  # Bx8
                # dp = torch.cholesky_solve(-dC_IC_ZNSSD, L)[..., 0]  # Bx8
                # Get the norms
                norms = dp_norm_vectorized_gpu(dp, xi)  # B
                # Update the parameters
                Wp = warp.W_vectorized_gpu(p)  # Bx3x3
                Wdp = warp.W_vectorized_gpu(dp)  # Bx3x3
                Wpdp = torch.matmul(Wp, torch.linalg.inv(Wdp))  # Bx3x3
                p = ((Wpdp / Wpdp[:, -1:, -1:]) - torch.eye(3)[None, ...])  # Bx3x3
                p = p.reshape(-1, 9)[:, :8]  # Bx8
                # Check for convergence
                if norms.max() < conv_tol:
                    break
            if self.verbose:
                for i in range(len(norms)):
                    t = np.squeeze(t[i].cpu().numpy())[self.subset_slice]
                    t = (t - t.mean()) / np.sqrt(((t - t.mean()) ** 2).sum())
                    t_d = np.squeeze(t_deformed[i].cpu().numpy())
                    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                    ax[0].imshow(np.squeeze(r.cpu().numpy()), cmap="gray")
                    ax[0].set_title("Reference")
                    ax[0].axis("off")
                    ax[1].imshow(t_d, cmap="gray")
                    ax[1].set_title("Deformed Target")
                    ax[1].axis("off")
                    ax[2].imshow(np.abs(e[i].cpu().numpy()).reshape(r.shape[2:]), cmap="gray")
                    ax[2].set_title("Residuals")
                    ax[2].axis("off")
                    ax[3].imshow(np.abs(t - t_d), cmap="gray")
                    plt.tight_layout()
                    plt.savefig(f"./gif/{idx[i]}_GPU.png")
                    plt.close("all")
            # print(f"Batch converged in {num_iter} iterations.")
            # Store the results
            residuals_out.append(residuals.cpu().numpy())
            norms_out.append(norms.cpu().numpy())
            num_iters_out.append(num_iter * np.ones_like(residuals.cpu().numpy()))
            homographies_out.append(p.cpu().numpy())
        # Store the results
        self.results.homographies[self.roi] = np.concatenate(homographies_out, axis=0)
        self.results.num_iter[self.roi] = np.concatenate(num_iters_out, axis=0)
        self.results.residuals[self.roi] = np.concatenate(residuals_out, axis=0)
        self.results.norms[self.roi] = np.concatenate(norms_out, axis=0)

    def save(self, filename: str) -> None:
        """Save the optimizer to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def load(self, filename: str) -> None:
        """Load the optimizer from a file."""
        with open(filename, "rb") as f:
            obj = dill.load(f)
        self.__dict__.update(obj.__dict__)

# Functions for the inverse composition gauss-newton algorithm


def dp_norm_vectorized_gpu(dp, xi) -> float:
    """Compute the norm of the delta p vector.
    Assumes dp is a (B, 8) array and xi is a (*, 2) array."""
    xi1max = xi[..., 0].max()
    xi2max = xi[..., 1].max()
    ximax = torch.tensor([[xi1max, xi2max]])  # Bx2
    dp_i0 = torch.square(dp[:, 0:2] * ximax).sum(axis=-1)
    dp_i1 = torch.square(dp[:, 3:5] * ximax).sum(axis=-1)
    dp_i2 = torch.square(dp[:, 6:8] * ximax).sum(axis=-1)
    out = torch.sqrt(dp_i0 + dp_i1 + dp_i2 + torch.square(dp[:, 2]) + torch.square(dp[:, 5]))
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


class Results:
    def __init__(self,
                 shape: tuple = None,
                 PC: ARRAY = None,
                 x0: ARRAY = None,
                 rel_step_size: NUMBER = None,
                 fixed_projection: bool = False,
                 detector_tilt: NUMBER = 10.0,
                 sample_tilt: NUMBER = 70.0,
                 traction_free: bool = True,
                 small_strain: bool = True,
                 C: ARRAY = None,
                 ):
        # Create scan details
        self.shape = shape
        self.yi, self.xi = np.indices(self.shape).astype(float)

        # Create xtal 2 sample transformation matrix
        x2s = np.array([180.0, 90 + sample_tilt - detector_tilt, 90.0], dtype=float)
        self.x2s = rotations.eu2om(np.deg2rad(x2s)).T

        # Create projection geometry
        if fixed_projection:
            self.PC_array = np.ones(shape + (3,), dtype=float) * PC
        elif rel_step_size is None or x0 is None:
            raise ValueError("The relative step size and the reference pattern location must be provided if the projection is not fixed.")
        else:
            theta = np.radians(90 - sample_tilt)
            phi = np.radians(detector_tilt)
            self.PC_array = np.array([
                PC[0] - (x0[1] - self.xi) * rel_step_size,
                PC[1] - (x0[0] - self.yi) * rel_step_size * np.cos(theta) / np.cos(phi),
                PC[2] - (x0[0] - self.yi) * rel_step_size * np.sin(theta + phi)
            ]).transpose(1, 2, 0)

        # Create calculation parameters
        if traction_free and C is None:
            raise ValueError("The stiffness tensor must be provided if the calculation is traction free.")
        self.traction_free = traction_free
        self.C = C
        self.small_strain = small_strain

        # Create empty arrays for the results
        self.num_iter = np.zeros(shape, dtype=int)
        self.residuals = np.zeros(shape, dtype=float)
        self.norms = np.zeros(shape, dtype=float)
        self.homographies = np.zeros(shape + (8,), dtype=float)
        self.strains = np.zeros(shape + (3, 3), dtype=float)
        self.rotations = np.zeros(shape + (3, 3), dtype=float)
        self.stresses = np.zeros(shape + (3, 3), dtype=float)
        self.F = np.zeros(shape + (3, 3), dtype=float)

    def save(self, filename: str) -> None:
        """Save the results to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def load(self, filename: str) -> None:
        """Load the results from a file."""
        with open(filename, "rb") as f:
            obj = dill.load(f)
            self.__dict__.update(obj.__dict__)

    def update(self, icgn_out: list, roi: np.ndarray) -> None:
        """Update the results with the output of the homography optimization."""
        # Get the results
        p_roi = np.array([o[0] for o in icgn_out])
        num_iter_roi = np.array([o[1] for o in icgn_out])
        residuals_roi = np.array([o[2] for o in icgn_out])
        norms_roi = np.array([o[3] for o in icgn_out])
        # Store the results
        self.homographies[roi] = p_roi
        self.num_iter[roi] = num_iter_roi
        self.residuals[roi] = residuals_roi
        self.norms[roi] = norms_roi

    def calculate(self, roi: np.ndarray = None, free_to_dilate=True) -> None:
        """Calculate the strains, rotations, and stresses from the homographies."""
        if roi is None:
            roi = np.s_[:]
        if not free_to_dilate:
            self.homographies[..., 0] = 1.0
            self.homographies[..., 4] = 1.0
        F_roi = conversions.h2F(self.homographies[roi], self.PC_array[roi])
        F_roi = np.matmul(self.x2s, np.matmul(F_roi, self.x2s.T))
        F_roi = F_roi / F_roi[..., 2, 2][..., None, None]
        self.F[roi] = F_roi
        calc_out = conversions.F2strain(self.F[roi], self.C, self.small_strain)
        self.strains[roi], self.rotations[roi] = calc_out[:2]
        if self.traction_free:
            self.stresses[roi] = calc_out[2]
