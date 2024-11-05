import time
import numpy as np
from scipy import interpolate
import torch

import matplotlib.pyplot as plt

import Data
import utilities
import warp
import bspline_gpu as gpu_warp

torch.set_default_device("cuda")


# Get the data
up2 = "E:/SiGe/a-C03-scan/ScanA_1024x1024.up2"
ang = "E:/SiGe/a-C03-scan/ScanA.ang"
pat_obj = Data.UP2(up2)
ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
im = pat_obj.read_pattern(0, process=False)
im = np.clip(im, np.percentile(im, 1), np.percentile(im, 99))
im = (im - np.min(im)) / (np.max(im) - np.min(im))

# Define a simple rotation matrix to rotate the image
theta = np.deg2rad(10)
t1 = 10
t2 = 10
h = np.array(
    [
        np.cos(theta) - 1,
        -np.sin(theta),
        t1 * np.cos(theta - t2 * np.sin(theta)),
        np.sin(theta),
        np.cos(theta) - 1,
        t1 * np.sin(theta) + t2 * np.cos(theta),
        0,
        0,
    ]
)
s = (slice(100, -100), slice(100, -100))

# Create coordinates
x = np.arange(im.shape[1])# - im.shape[1] / 2
y = np.arange(im.shape[0])# - im.shape[0] / 2
X, Y = np.meshgrid(x, y)
xi = np.array([Y[s].flatten(), X[s].flatten()])
xi_prime = warp.get_xi_prime(xi, h)

print("IM shape", im.shape, xi.shape, xi_prime.shape)
# Create the CPU spline
t0 = time.time()
spline = interpolate.RectBivariateSpline(x, y, im, kx=5, ky=5)
im_prime = spline(xi_prime[0], xi_prime[1], grid=False).reshape(im[s].shape)
GRx = spline(xi[0], xi[1], dx=1, dy=0, grid=False).reshape(im[s].shape)
GRy = spline(xi[0], xi[1], dx=0, dy=1, grid=False).reshape(im[s].shape)
GR = np.sqrt(GRx ** 2 + GRy ** 2)
print("CPU spline time:", time.time() - t0)


# Create the GPU spline
interpolation = [5, 5]
bound = [0, 0]
extrapolate = True

# Convert to GPU tensors
im_gpu = torch.tensor(im).float().cuda().reshape(1, 1, *im.shape)
x = torch.arange(im.shape[1])# - im.shape[1] / 2
y = torch.arange(im.shape[0])# - im.shape[0] / 2
X, Y = torch.meshgrid(x, y, indexing="xy")
xi = torch.stack([Y[s], X[s]], dim=-1).unsqueeze(0)  # 1xHxWx2
h = torch.tensor(h).float().cuda().unsqueeze(0)  # 1x8
xi_prime = warp.get_xi_prime_vectorized_gpu(xi, h)

print("IM shape", im_gpu.shape, xi.shape, xi_prime.shape)

# Convert xi
t0 = time.time()
bound = gpu_warp.pad_list_int(bound, xi_prime.shape[-1])
interpolation = gpu_warp.pad_list_int(interpolation, xi_prime.shape[-1])
bound_fn = gpu_warp.make_bound(bound)
spline_fn = gpu_warp.make_spline(interpolation)
im_prime_gpu = gpu_warp.pull(im_gpu, xi_prime, bound_fn, spline_fn, extrapolate)
GR_gpu = gpu_warp.grad(im_gpu, xi, bound_fn, spline_fn, extrapolate)
im_prime_gpu = np.squeeze(im_prime_gpu.cpu().numpy())
GR_gpu = np.squeeze(GR_gpu.cpu().numpy())
GR_gpu = np.sqrt(GR_gpu[..., 0] ** 2 + GR_gpu[..., 1] ** 2)
print("GPU spline time:", time.time() - t0)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0, 0].imshow(im_prime, cmap="gray")
ax[0, 0].set_title("CPU")
ax[0, 1].imshow(im_prime_gpu, cmap="gray")
ax[0, 1].set_title("GPU")
ax[1, 0].imshow(GR, cmap="gray")
ax[1, 0].set_title("CPU")
ax[1, 1].imshow(GR_gpu, cmap="gray")
ax[1, 1].set_title("GPU")
plt.tight_layout()
plt.show()



exit()
def run_gpu(self, max_iter=50, conv_tol=1e-3, batch_size=8, verbose=False) -> np.ndarray:
    """Run the homography optimization."""
    # Precompute the reference subset
    indices = self.pidx[self.roi]
    self.reference_precompute()
    self.max_iter = max_iter
    self.conv_tol = conv_tol
    self.verbose = verbose
    # Run the optimization
    print("Running the homography optimization...")
    # Split indices into batches
    indices_split = np.array_split(indices, len(indices) // batch_size)
    # Create spline stuff
    bound_fn = gpu_warp.make_bound([0, 0])
    spline_fn = gpu_warp.make_spline([5, 5])
    # Loop over the batches
    for indices in indices_split:
        # Get the target images and process them
        T = self.pat_obj.read_pattern(indices, process=True, p_kwargs=self.image_processing_kwargs)
        # Run homography initialization
        if self.init_type is None or self.init_type == "none":
            p = torch.zeros((len(indices), 8), dtype=torch.float32).cuda()
        else:
            raise NotImplementedError("GPU initialization not implemented yet!")
        # Run the optimization
        num_iter = 0
        while num_iter < self.max_iter:
            # Warp the target subset
            num_iter += 1
            xi_prime_gpu = ...
            T_deformed = gpu_warp.pull(im_gpu, xi_prime_gpu, bound_fn, spline_fn, extrapolate=False)
            # Normalize the target
            T_mean = T_deformed.mean()
            T_deformed = (T_deformed - T_mean) / torch.sqrt(((T_deformed - T_mean) ** 2).sum())
            # Compute the residuals
            e = self.icgn_pre.r - T_deformed
            residuals = torch.abs(e).mean(dim=(1, 2, 3))  # Take residuals over the C, H, W dimensions, leave the batch dimension
            # Compute the gradient of the correlation criterion
            dC_IC_ZNSSD = 2 / self.icgn_pre.r_zmsv * torch.einsum('ij,kj->ik', e, self.icgn_pre.NablaR_dot_Jac)  # 8 x N
            dp = torch.cholesky_solve(-dC_IC_ZNSSD.T, self.icgn_pre.L).T  # N x 8
            # Get the norms
            norms = dp_norm_vectorized_gpu(dp, xi)  # N
            # Update the parameters
            Wp = W_vectorized_gpu(p)  # N x 3 x 3
            Wdp = W_vectorized_gpu(dp)  # N x 3 x 3
            Wpdp = Wp @ torch.linalg.inv(Wdp)  # N x 3 x 3
            p = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - torch.eye(3)[None, ...])  # N x 3 x 3
            p = p.reshape(-1, 9)[:, :8]  # N x 8
            # Update the active targets
            # print(f"Iteration {num_iter.max()}: Number of active targets: {active.sum()}")
            if norms.max() < conv_tol:
                break

    
    # Store the results
    self.results.update(out, self.roi)


def run_single(idx: int) -> np.ndarray:
    """Run the homography optimization for a single point."""
    # Get the target image and process it
    # T = utilities.get_pattern(self.pat_obj, idx)
    # T = utilities.process_pattern(T, **self.image_processing_kwargs)
    T = self.pat_obj.read_pattern(idx, process=True, p_kwargs=self.image_processing_kwargs)
    T_spline = interpolate.RectBivariateSpline(
        self.icgn_pre.x, self.icgn_pre.y, T, kx=5, ky=5
        # self.precomputed.x, self.precomputed.y, normalize(T), kx=5, ky=5
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
        if self.extra_verbose:
            print(f"Pattern {idx}: Iteration {num_iter}, Norm: {norm:.4f}, Residual: {residuals[-1]:.4f}")
            row = int(self.icgn_pre.xi[0].max() - self.icgn_pre.xi[0].min() + 1)
            col = int(self.icgn_pre.xi[1].max() - self.icgn_pre.xi[1].min() + 1)
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(self.icgn_pre.r.reshape(row, col), cmap="gray")
            ax[0].set_title("Reference")
            ax[1].imshow(t_deformed.reshape(row, col), cmap="gray")
            ax[1].set_title("Deformed target")
            ax[2].imshow(e.reshape(row, col), cmap="gray", vmin=-0.001, vmax=0.001)
            ax[2].set_title("Final residual")
            for a in ax.flatten():
                a.axis("off")
            plt.tight_layout()
            plt.savefig(f"gif/IC-GN_{idx}_{num_iter}.png")
            plt.close(fig)
        if norm < self.conv_tol:
            break
    if self.verbose:
    # if num_iter >= self.max_iter and self.verbose:
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