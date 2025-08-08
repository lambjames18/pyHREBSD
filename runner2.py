import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import Data
import utilities
import get_homography_cpu as core

name = "SiGe_EDAX"
up2 = "E:/SiGe_edax/10 mu squares map no averaging3a_spherical.up2"
ang = "E:/SiGe_edax/10 mu squares map no averaging3a_spherical.ang"
x0 = (80, 101)

pat_obj = Data.UP2(up2)
ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print(x0)

shape = (161, 202)
calc = True


if calc:
    h, iterations, residuals, dp_norms = core.optimize(
        pat_obj, x0, crop_fraction=0.9, max_iter=50, n_jobs=16, verbose=True
    )

    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_homographies.npy",
        h,
    )
    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_iterations.npy",
        iterations,
    )
    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_residuals.npy",
        residuals,
    )
    np.save(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_dp_norms.npy",
        dp_norms,
    )
else:
    h = np.load(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_homographies.npy"
    )
    iterations = np.load(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_iterations.npy"
    )
    residuals = np.load(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_residuals.npy"
    )
    dp_norms = np.load(
        "E:/SiGe_edax/10 mu squares map no averaging3a_spherical_dp_norms.npy"
    )

h = h.reshape((shape[0], shape[1], 8))
iterations = iterations.reshape(shape)
residuals = residuals.reshape(shape)
dp_norms = dp_norms.reshape(shape)

io.imsave("E:/SiGe_edax/h11.tiff", h[:, :, 0].astype(np.float32))
io.imsave("E:/SiGe_edax/h12.tiff", h[:, :, 1].astype(np.float32))
io.imsave("E:/SiGe_edax/h13.tiff", h[:, :, 2].astype(np.float32))
io.imsave("E:/SiGe_edax/h21.tiff", h[:, :, 3].astype(np.float32))
io.imsave("E:/SiGe_edax/h22.tiff", h[:, :, 4].astype(np.float32))
io.imsave("E:/SiGe_edax/h23.tiff", h[:, :, 5].astype(np.float32))
io.imsave("E:/SiGe_edax/h31.tiff", h[:, :, 6].astype(np.float32))
io.imsave("E:/SiGe_edax/h32.tiff", h[:, :, 7].astype(np.float32))
io.imsave("E:/SiGe_edax/iterations.tiff", iterations.astype(np.uint8))
io.imsave("E:/SiGe_edax/residuals.tiff", residuals.astype(np.float32))
io.imsave("E:/SiGe_edax/dp_norms.tiff", dp_norms.astype(np.float32))

fig, ax = plt.subplots(4, 3, figsize=(15, 10))
ax[0, 0].imshow(iterations)
ax[0, 1].imshow(residuals)
ax[0, 2].imshow(dp_norms)
ax[1, 0].imshow(h[:, :, 0], cmap="gray")
ax[1, 1].imshow(h[:, :, 1], cmap="gray")
ax[1, 2].imshow(h[:, :, 2], cmap="gray")
ax[2, 0].imshow(h[:, :, 3], cmap="gray")
ax[2, 1].imshow(h[:, :, 4], cmap="gray")
ax[2, 2].imshow(h[:, :, 5], cmap="gray")
ax[3, 0].imshow(h[:, :, 6], cmap="gray")
ax[3, 1].imshow(h[:, :, 7], cmap="gray")
names = [
    "Iterations",
    "Residuals",
    "DP Norms",
    "H11",
    "H12",
    "H13",
    "H21",
    "H22",
    "H23",
    "H31",
    "H32",
    "H33",
]
for a in ax.ravel():
    a.axis("off")
    a.set_title(names.pop(0))

plt.tight_layout()
plt.show()
