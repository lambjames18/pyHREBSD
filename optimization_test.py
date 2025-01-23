# 3rd party imports
import numpy as np
from skimage import io
from scipy import linalg, interpolate, signal
import matplotlib.pyplot as plt

# Local imports
import warp
from get_homography_cpu import dp_norm, window_and_normalize, FMT
import conversions
from Data import process_pattern

np.set_printoptions(
    linewidth=125,
    precision=4,
    suppress=True,
    threshold=1000,
    formatter=None)

init_type = "partial"  # Type of initial guess: "partial" or "full"
max_iter = 50  # Maximum number of iterations
conv_tol = 5e-4  # Convergence tolerance
subset_size = 300  # Size of the subset cropped out from the center of the images for the optimization
target_path = "/Users/jameslamb/Downloads/deformed.jpeg"
reference_path = "/Users/jameslamb/Downloads/reference.jpeg"


#####################################################
# Pattern IO, setting up geometry, etc.
#####################################################

# Load the images
T = io.imread(target_path).astype(float)
R = io.imread(reference_path)
T = process_pattern(T, 0.0, 101, 3.0)
R = process_pattern(R, 0.0, 101, 0.0)
print(f"Reference - Min: {R.min()}, Max: {R.max()}, Mean: {R.mean()}, Shape: {R.shape}")
print(f"Target - Min: {T.min()}, Max: {T.max()}, Mean: {T.mean()}, Shape: {T.shape}")

# Rescale intensity to [0, 1]
T = (T - T.min()) / (T.max() - T.min())
R = (R - R.min()) / (R.max() - R.min())
print(f"Reference (rescaled) - Min: {R.min()}, Max: {R.max()}, Mean: {R.mean()}, Shape: {R.shape}")
print(f"Target (rescaled) - Min: {T.min()}, Max: {T.max()}, Mean: {T.mean()}, Shape: {T.shape}")

# Set the homography center and create the subset slice that we use to crop the images
h0 = (R.shape[1] / 2, R.shape[0] / 2)
r0 = int(max(0, h0[1] - subset_size // 2))
r1 = int(min(R.shape[0], h0[1] + subset_size // 2))
c0 = int(max(0, h0[0] - subset_size // 2))
c1 = int(min(R.shape[1], h0[0] + subset_size // 2))
subset_slice = (slice(r0, r1), slice(c0, c1))

# Create coordinate grid
x = np.arange(R.shape[1]) - h0[0]
y = np.arange(R.shape[0]) - h0[1]
print(f"X: {x.shape}, Y: {y.shape}")
X, Y = np.meshgrid(x, y, indexing="xy")
xi = np.array([Y[subset_slice].flatten(), X[subset_slice].flatten()])

#####################################################
# Reference precomputing
#####################################################

# Fit the spline to the unormalized reference image
R_spline = interpolate.RectBivariateSpline(y, x, R, kx=5, ky=5)

# Normalize the reference image
r = R_spline(xi[0], xi[1], grid=False).flatten()
r_zmsv = np.sqrt(((r - r.mean()) ** 2).sum())
r = (r - r.mean()) / r_zmsv
print(f"Reference normalized - Min: {r.min()}, Max: {r.max()}, Mean: {r.mean()}, Shape: {r.shape}")
print(f"R_zmsv: {r_zmsv}")

# Create gradients
GRx = R_spline(xi[0], xi[1], dx=0, dy=1, grid=False)
GRy = R_spline(xi[0], xi[1], dx=1, dy=0, grid=False)
# GRy, GRx = np.gradient(R[subset_slice], axis=(0, 1))
print(f"Gradients (x) - Min: {GRx.min()}, Max: {GRx.max()}, Mean: {GRx.mean()}, Shape: {GRx.shape}")
print(f"Gradients (y) - Min: {GRy.min()}, Max: {GRy.max()}, Mean: {GRy.mean()}, Shape: {GRy.shape}")
GR = np.vstack((GRy, GRx)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN -> 1x2xN

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for a in ax.ravel():
    a.axis("off")
ax[0].imshow(GRx.reshape(subset_size, subset_size), cmap="Greys_r")
ax[0].set_title("Gradient (x)")
ax[1].imshow(GRy.reshape(subset_size, subset_size), cmap="Greys_r")
ax[1].set_title("Gradient (y)")
plt.tight_layout()
plt.savefig("gradients.jpg")
plt.close()

# Compute the jacobian of the shape function
_1 = np.ones(xi.shape[1])
_0 = np.zeros(xi.shape[1])
out0 = np.array([[xi[0], xi[1], _1, _0, _0, _0, -xi[0] ** 2, -xi[1] * xi[0]]])
out1 = np.array([[_0, _0, _0, xi[0], xi[1], _1, -xi[0] * xi[1], -xi[1] ** 2]])
Jac = np.vstack((out0, out1))  # 2x8xN
print(f"Jacobian - Min: {Jac.min():.5f}, Max: {Jac.max():.5f}, Mean: {Jac.mean():.5f}, Shape: {Jac.shape}")

# Multiply the gradients by the jacobian
NablaR_dot_Jac = np.einsum("ilk,ljk->ijk", GR, Jac)[0]  # 1x8xN -> 8xN
print(f"NablaR_dot_Jac - Min: {NablaR_dot_Jac.min():.5f}, Max: {NablaR_dot_Jac.max():.5f}, Mean: {NablaR_dot_Jac.mean():.5f}, Shape: {NablaR_dot_Jac.shape}")
H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)
print("Hessian")
print(H)

# Compute the Cholesky decomposition of the Hessian
(c, L) = linalg.cho_factor(H)

# Compute initial guess reference stuff
guess_subset_slice = (slice(int(h0[1] - 128), int(h0[1] + 128)), slice(int(h0[0] - 128), int(h0[0] + 128)))
# Get the FMT-FCC initial guess precomputed items
r_init = window_and_normalize(R[guess_subset_slice])
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

#####################################################
# Initial guess
#####################################################

# Window and normalize the target
t_init = window_and_normalize(T[guess_subset_slice])
# Do the angle search first
t_init_fft = np.fft.fftshift(np.fft.fft2(t_init))
t_init_FMT, _ = FMT(t_init_fft, X_fmt, Y_fmt, x_fmt, y_fmt)
cc = signal.fftconvolve(r_fmt, t_init_FMT[::-1], mode="same").real
theta = (np.argmax(cc) - len(cc) / 2) * np.pi / len(cc)
# Apply the rotation
h = conversions.xyt2h_partial(np.array([[0, 0, -theta]]))[0]
t_init_rot = warp.deform_image(t_init, h, h0)
# Do the translation search
cc = signal.fftconvolve(
    r_init, t_init_rot[::-1, ::-1], mode="same"
).real
shift = np.unravel_index(np.argmax(cc), cc.shape) - np.array(cc.shape) / 2

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a in ax.ravel():
    a.axis("off")
ax[0].imshow(r_init, cmap="Greys_r", vmin=r_init.min(), vmax=r_init.max())
ax[0].set_title("Reference")
ax[1].imshow(t_init, cmap="Greys_r", vmin=r_init.min(), vmax=r_init.max())
ax[1].set_title("Target")
ax[2].imshow(cc, cmap="Greys_r")
ax[2].set_title("Cross-correlation")
ax[2].scatter(shift[1] + cc.shape[1] / 2, shift[0] + cc.shape[0] / 2, c="r", marker="+")
ax[2].scatter(cc.shape[1] / 2, cc.shape[0] / 2, c="b", marker="x")
plt.tight_layout()
plt.show()
# plt.savefig("initial_guess.jpg")
# plt.close()

# Store the homography
measurement = np.array([[-shift[0], -shift[1], -theta]])
# Convert the measurements to homographies
if init_type == "full":
    p = conversions.xyt2h(measurement, h0)
else:
    p = conversions.xyt2h_partial(measurement)

#####################################################
# IC-GN algorithm
#####################################################

# Fit the spline to the unormalized deformed image
T_spline = interpolate.RectBivariateSpline(y, x, T, kx=5, ky=5)

p = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 3.0, 0.0, 0.0])

# Run the optimization
num_iter = 0
norms = []
residuals = []
print("Beginning optimization...")
print(f"Initial guess: {p}")
while num_iter < max_iter:
    print(f"Iteration {num_iter}")

    # Warp the target subset
    num_iter += 1
    t_deformed = warp.deform(xi, T_spline, p)
    t_mean = t_deformed.mean()
    print(f"  - (Deformed) Min: {t_deformed.min():.5f}, Max: {t_deformed.max():.5f}")
    t_deformed = (t_deformed - t_mean) / np.sqrt(((t_deformed - t_mean) ** 2).sum())
    print(f"  - (Deformed normalized) Min: {t_deformed.min():.5f}, Max: {t_deformed.max():.5f}")

    # Compute the residuals
    e = r - t_deformed
    residuals.append(np.abs(e).mean())
    print(f"  - (Residual) Min: {e.min():.5f}, Max: {e.max():.5f}, Mean: {e.mean():.5f}")

    # Copmute the gradient of the correlation criterion
    dC_IC_ZNSSD = (2 / r_zmsv * np.matmul(e, NablaR_dot_Jac.T))  # 8x1
    print(f"  - (dC_IC_ZNSSD)", dC_IC_ZNSSD)

    # Find the deformation incriment, delta_p, by solving the linear system
    dp = linalg.cho_solve((c, L), -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]
    print(f"  - (dp)", dp)

    # Update the parameters
    norm = dp_norm(dp, xi)
    print(f"  - (norm)", norm)
    Wp = warp.W(p)
    Wdp = warp.W(dp)
    Wpdp = np.matmul(Wp, np.linalg.inv(Wdp))
    p = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).reshape(9)[:8]
    print(f"  - (p)", p)

    # Store the update
    norms.append(norm)
    # print(f"Pattern {idx}: Iteration {num_iter}, Norm: {norm:.4f}, Residual: {residuals[-1]:.4f}")
    if norm < conv_tol:
        break

fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
for a in ax.ravel():
    a.axis("off")

# Compute the initial deformed image
t_deformed = warp.deform(xi, T_spline, np.zeros(8))
t_mean = t_deformed.mean()
t_deformed = (t_deformed - t_mean) / np.sqrt(((t_deformed - t_mean) ** 2).sum())
r = r.reshape(subset_size, subset_size)
t_deformed = t_deformed.reshape(subset_size, subset_size)
res = r - t_deformed
vmin = res.min()
vmax = res.max()
ax[0, 0].imshow(r, cmap="Greys_r")
ax[0, 0].set_title("Reference")
ax[0, 1].imshow(t_deformed, cmap="Greys_r")
ax[0, 1].set_title("Deformed initial")
ax[0, 2].imshow(res, cmap="Greys_r",vmin=vmin, vmax=vmax)
ax[0, 2].set_title("Residuals")

# Compute the final deformed image
t_deformed = warp.deform(xi, T_spline, p)
t_mean = t_deformed.mean()
t_deformed = (t_deformed - t_mean) / np.sqrt(((t_deformed - t_mean) ** 2).sum())
t_deformed = t_deformed.reshape(subset_size, subset_size)
res = r - t_deformed
ax[1, 0].imshow(r, cmap="Greys_r")
ax[1, 0].set_title("Reference")
ax[1, 1].imshow(t_deformed, cmap="Greys_r")
ax[1, 1].set_title("Deformed warped")
ax[1, 2].imshow(res, cmap="Greys_r")#, vmin=vmin, vmax=vmax)
ax[1, 2].set_title("Residuals")

plt.tight_layout()
plt.savefig("results.jpg")
plt.close()
