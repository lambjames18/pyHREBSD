import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities
import get_homography_cpu as core

name = "GaN27238_512"
up2 = (
    "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_256x256_flipX.up2"
)
# up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
ang = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_flipX.ang"
x0 = (100, 100)

pat_obj = Data.UP2(up2)
ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
x0 = np.ravel_multi_index(x0, ang_data.shape)
print(x0)

h, iterations, residuals, dp_norms = core.optimize(
    pat_obj, x0, crop_fraction=0.9, max_iter=25, n_jobs=19, verbose=True
)

np.save(
    "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_256x256_flipX_homographies.npy",
    h,
)
np.save(
    "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_256x256_flipX_iterations.npy",
    iterations,
)
np.save(
    "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_256x256_flipX_residuals.npy",
    residuals,
)
np.save(
    "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_256x256_flipX_dp_norms.npy",
    dp_norms,
)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(iterations)
ax[1].imshow(residuals)
ax[2].imshow(dp_norms)
plt.tight_layout()
plt.show()
