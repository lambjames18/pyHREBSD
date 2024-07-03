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

# Create coordinates
x = np.arange(im.shape[1]) - im.shape[1] / 2
y = np.arange(im.shape[0]) - im.shape[0] / 2
X, Y = np.meshgrid(x, y)
xi = np.array([Y.flatten(), X.flatten()])
xi_prime = warp.get_xi_prime(xi, h)

# Create the CPU spline
spline = interpolate.RectBivariateSpline(x, y, im, kx=5, ky=5)
im_prime = spline(xi_prime[0], xi_prime[1], grid=False).reshape(im.shape)

# Create the GPU spline
interpolation = [5, 5]
bound = [1, 1]
extrapolate = True

# Convert to GPU tensors
im_gpu = torch.tensor(im).float().cuda().reshape(1, 1, *im.shape)
xi_gpu = torch.tensor(xi.T).float().cuda().reshape(1, *im.shape, 2)
xi_prime_gpu = torch.tensor(xi_prime.T).float().cuda().reshape(1, *im.shape, 2)

# Convert xi
xi_gpu += im.shape[0] / 2
xi_prime_gpu += im.shape[0] / 2
im_prime_gpu = gpu_warp.grid_pull(im_gpu, xi_prime_gpu, bound, interpolation, extrapolate)
im_prime_gpu = np.squeeze(im_prime_gpu.cpu().numpy())
print(im_prime_gpu.shape)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0, 0].imshow(im, cmap="gray")
ax[0, 0].set_title("Original")
ax[0, 1].imshow(im_prime, cmap="gray")
ax[0, 1].set_title("CPU")
ax[1, 0].imshow(im, cmap="gray")
ax[1, 0].set_title("Original")
ax[1, 1].imshow(im_prime_gpu, cmap="gray")
ax[1, 1].set_title("GPU")
plt.tight_layout()
plt.show()
