import numpy as np
import matplotlib.pyplot as plt

import utilities

shape = (200, 200)
PC = (256.0, 256.0, 409.6)
fixed_projection = True
x0 = (100, 100)
pixel_size = 26.0  # The pixel size in um, taking binning into account (so 4xpixel_size for 4x4 binning)
sample_tilt = 70.0  # The sample tilt in degrees
detector_tilt = 8.5  # The detector tilt in degrees
step_size = 0.02  # The step size in um
traction_free = True
small_strain = False
C = utilities.get_stiffness_tensor(365.0, 135.0, 114.0, 381.0, 109.0, structure="hexagonal")

results = utilities.Results(shape, PC, x0, step_size / pixel_size, fixed_projection, detector_tilt, sample_tilt, traction_free, small_strain, C)
results.load(f"results/GaN27238_results.pkl")
results.shape = shape
results.C = C
results.PC_array = np.ones(shape + (3,), dtype=float) * PC
results.traction_free = traction_free
results.small_strain = small_strain
results.calculate()
h_256 = results.homographies.copy()
iter_256 = results.num_iter.copy()

results = utilities.Results(shape, PC, x0, step_size / pixel_size, fixed_projection, detector_tilt, sample_tilt, traction_free, small_strain, C)
results.load(f"results/GaN27238_512_results.pkl")
results.shape = shape
results.C = C
results.PC_array = np.ones(shape + (3,), dtype=float) * PC
results.traction_free = traction_free
results.small_strain = small_strain
results.calculate()
h_512 = results.homographies.copy()
iter_512 = results.num_iter.copy()

residual = np.abs(h_512 - h_256)

fig, ax = plt.subplots(3, 3, figsize=(8, 8))
ax = ax.ravel()
for i in range(8):
    im = ax[i].imshow(residual[..., i], cmap="Greys_r")
    ax[i].scatter(x0[1], x0[0], c="r", s=10, marker="x")
    ax[i].axis("off")
ax[-1].axis("off")
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
plt.show()

max_iter = max(iter_256.max(), iter_512.max())

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.hist(iter_256.flatten(), bins=max_iter, range=(0, max_iter), alpha=0.6, label="256", color="#ceac5c")
ax.hist(iter_512.flatten(), bins=max_iter, range=(0, max_iter), alpha=0.6, label="512", color="#165b33")
ax.set_xlabel("Number of iterations", fontsize=20)
ax.set_ylabel("Number of pixels", fontsize=20)
ax.legend()
utilities.standardize_axis(ax, labelsize=18)
ax.set_xticks(np.arange(0, max_iter, 2))
ax.set_xticklabels(np.arange(0, max_iter, 2), fontsize=18)
plt.tight_layout()
plt.show()

