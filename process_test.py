import timeit
import numpy as np
from skimage import io, exposure
from scipy import ndimage
import torch
import kornia


def process_pattern(img: np.ndarray, sigma: float = 0.0, equalize: bool = True, truncate: bool = True) -> np.ndarray:
    """Cleans patterns by equalizing the histogram and normalizing.

    Args:
        img (np.ndarray): The patterns to clean. (H, W)
        equalize (bool): Whether to equalize the histogram.
        high_pass (bool): Whether to apply a high-pass filter.
        truncate (bool): Whether to truncate the patterns.

    Returns:
        np.ndarray: The cleaned patterns. (N, H, W)"""
    # Process inputs
    img = img.astype(np.float32).reshape(1, 1, img.shape[0], img.shape[1])
    median = np.percentile(img, 50)
    low_percentile = np.percentile(img, 1)
    high_percentile = np.percentile(img, 99)

    # Convert to torch tensor, set device, create output tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.tensor(img, dtype=torch.float32).to("cpu")

    # Create processing functions
    def get_kernel(sigma):
        k = sigma if sigma % 2 == 1 else sigma + 1
        return (int(k), int(k)), (float(sigma), float(sigma))

    # Process the patterns, using batches
    bk, bs = get_kernel(int(img.shape[-1] / 10))
    if truncate:
        img[img < low_percentile] = median
        img[img > high_percentile] = median
    img = kornia.enhance.normalize_min_max(img, 0.0, 1.0)
    background = kornia.filters.gaussian_blur2d(img, bk, bs)
    img = img - background
    img = kornia.enhance.normalize_min_max(img, 0.0, 1.0)
    if equalize:
        img = kornia.enhance.equalize_clahe(img)
        img = kornia.enhance.normalize_min_max(img, 0.0, 1.0)
    if sigma > 0.0:
        k, s = get_kernel(sigma)
        img = kornia.filters.gaussian_blur2d(img, k, s)
        img = kornia.enhance.normalize_min_max(img, 0.0, 1.0)

    out = img.cpu().numpy()
    out = out.reshape(out.shape[2], out.shape[3])
    torch.cuda.empty_cache()
    return out



def process_pattern_numpy(img: np.ndarray, sigma: float = 0.0, equalize: bool = True, truncate: bool = True) -> np.ndarray:
    """Cleans patterns by equalizing the histogram and normalizing.

    Args:
        img (np.ndarray): The patterns to clean. (H, W)
        equalize (bool): Whether to equalize the histogram.
        high_pass (bool): Whether to apply a high-pass filter.
        truncate (bool): Whether to truncate the patterns.

    Returns:
        np.ndarray: The cleaned patterns. (N, H, W)"""
    # Process inputs
    img = img.astype(np.float32)
    median = np.percentile(img, 50)
    low_percentile = np.percentile(img, 1)
    high_percentile = np.percentile(img, 99)

    # Process the patterns
    if truncate:
        img[img < low_percentile] = median
        img[img > high_percentile] = median
    img = (img - img.min()) / (img.max() - img.min())
    background = ndimage.gaussian_filter(img.mean(axis=0), img.shape[-1] / 10)
    img = img - background
    img = (img - img.min()) / (img.max() - img.min())
    if equalize:
        img = exposure.equalize_adapthist(img)
        img = (img - img.min()) / (img.max() - img.min())
    if sigma > 0.0:
        img = ndimage.gaussian_filter(img, sigma)
        img = (img - img.min()) / (img.max() - img.min())

    return img

path = "/Users/jameslamb/Library/Mobile Documents/com~apple~CloudDocs/Downloads/CoNi67_Pattern.tif"
img = io.imread(path)

sigma = 10.0
equalize = True
truncate = True

print("Kornia")
print(timeit.timeit(lambda: process_pattern(img, sigma, equalize, truncate), number=100) / 100)
print("Numpy")
print(timeit.timeit(lambda: process_pattern_numpy(img, sigma, equalize, truncate), number=100) / 100)
