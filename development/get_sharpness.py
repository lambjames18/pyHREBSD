from collections import namedtuple
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage import io
import mpire
import torch

import Data
import utilities


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def _get_sharpness_torch(ebsd_data: namedtuple, batch_size: int = 8, lazy=True) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.

    Args:
        imgs (np.ndarray): The images to calculate the sharpness of. (H, W) or (N, H, W) or (N0, N1, H, W)

    Returns:
        np.ndarray: The sharpness of the images. float, (N,) or (N0, N1) for example."""
    # Process inputs
    if not lazy:
        imgs = ebsd_data.get_patterns()

        # Convert to torch tensor, set device, create output tensor
        imgs = torch.tensor(imgs, dtype=torch.float32).to(device)  # (N, H, W)

        # Separate the imgs into batches
        imgs_split = list(torch.split(imgs, batch_size, dim=0))  # (M, batch_size, H, W) where M is the number of batches and batch_size is not a constant
        shp = torch.tensor([], dtype=torch.float32).to(device)

        # Calculate sharpness
        for i in tqdm(range(len(imgs_split)), desc='Calculating sharpness', unit='batches'):
            shp = torch.cat((shp, _calc_sharpness_torch(imgs_split[i])))

    else:
        # Put into batches
        idx = np.arange(ebsd_data.nPatterns)
        idx_split = np.array_split(idx, ebsd_data.nPatterns // batch_size)  # (M, batch_size) where M is the number of batches and batch_size is not a constant
        args = [(ebsd_data, idx) for idx in idx_split]

        # shp = torch.tensor([], dtype=torch.float32).to(device)
        shp = np.array([])
        for i in tqdm(range(len(idx_split)), desc='Calculating sharpness', unit='batches'):
            shp = np.concatenate((shp, _calc_sharpness_torch_lazy(args[i]).cpu().numpy()))

    # Convert to numpy and reshape if necessary
    shp = np.squeeze(shp)
    return shp


def _calc_sharpness_torch_lazy(args: tuple) -> torch.Tensor:
    """Calculates the sharpness of an image/stack of images.

    Args:
        args (tuple): The arguments to calculate the sharpness of. (ebsd_data, idx)

    Returns:
        torch.Tensor: The sharpness of the images. float, (N,)"""
    ebsd_data, idx = args
    pats = ebsd_data.get_patterns(idx)
    pats = torch.tensor(pats, dtype=torch.float32).to(device)
    f = torch.fft.fft2(pats)
    f = torch.real(f)
    fshift = torch.fft.fftshift(f)
    AF = torch.abs(fshift)
    thresh = torch.amax(AF, dim=(1, 2), keepdim=True) / 2500
    th = torch.sum(fshift > thresh, dim=(1, 2))
    return th / (pats.shape[1] * pats.shape[2])


def _calc_sharpness_torch(imgs: torch.Tensor) -> torch.Tensor:
    """Calculates the sharpness of an image/stack of images.

    Args:
        imgs (torch.Tensor): The images to calculate the sharpness of. (N, 1, H, W)

    Returns:
        torch.Tensor: The sharpness of the images. float, (N,)"""
    f = torch.fft.fft2(imgs)
    f = torch.real(f)
    fshift = torch.fft.fftshift(f)
    AF = torch.abs(fshift)
    thresh = torch.amax(AF, dim=(1, 2), keepdim=True) / 2500
    th = torch.sum(fshift > thresh, dim=(1, 2), keepdim=True)
    return th / (imgs.shape[1] * imgs.shape[2])


def _get_sharpness_numpy(ebsd_data: namedtuple, batch_size: int = 8, parallel: bool = False, lazy = True) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.

    Args:
        imgs (np.ndarray): The images to calculate the sharpness of. (H, W) or (N, H, W) or (N0, N1, H, W)
        batch_size (int): The number of images to process at once.

    Returns:
        np.ndarray: The sharpness of the images. float, (N,) or (N0, N1) for example."""
    if not lazy:
        imgs = ebsd_data.get_patterns()
        # Put into batches
        imgs_split = np.array_split(imgs, imgs.shape[0] // batch_size)  # (M, batch_size, H, W) where M is the number of batches and batch_size is not a constant

        if parallel:
            with mpire.WorkerPool(n_jobs=os.cpu_count() // 2) as pool:
                shp = pool.map(_calc_sharpness_numpy, imgs_split, progress_bar=True)
        else:
            shp = np.array([])
            for i in tqdm(range(len(imgs_split)), desc='Calculating sharpness', unit='batches'):
                shp = np.concatenate((shp, _calc_sharpness_numpy(imgs_split[i])))
    else:
        # Put into batches
        idx = np.arange(ebsd_data.nPatterns)
        idx_split = np.array_split(idx, ebsd_data.nPatterns // batch_size)  # (M, batch_size) where M is the number of batches and batch_size is not a constant
        args = [(ebsd_data, idx) for idx in idx_split]

        if parallel:
            with mpire.WorkerPool(n_jobs=os.cpu_count() // 2) as pool:
                shp = pool.map(_calc_sharpness_numpy_lazy, args, progress_bar=True)
        else:
            shp = np.array([])
            for i in tqdm(range(len(idx_split)), desc='Calculating sharpness', unit='batches'):
                shp = np.concatenate((shp, _calc_sharpness_numpy_lazy(args[i])))

    # Reshape if necessary
    shp = np.squeeze(shp)
    return shp


def _calc_sharpness_numpy_lazy(args: tuple) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.

    Args:
        args (tuple): The arguments to calculate the sharpness of. (ebsd_data, idx)

    Returns:
        np.ndarray: The sharpness of the images. float, (N,)"""
    ebsd_data, idx = args
    pats = ebsd_data.get_patterns(idx)
    f = np.fft.fft2(pats, axes=(1,2))
    f = np.real(f)
    fshift = np.fft.fftshift(f, axes=(1,2))
    AF = abs(fshift)
    thresh = AF.max(axis=(1,2)) / 2500
    th = (fshift > thresh[:,None,None]).sum(axis=(1,2))
    return th / (pats.shape[1] * pats.shape[2])


def _calc_sharpness_numpy(imgs: np.ndarray) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.

    Args:
        imgs (np.ndarray): The images to calculate the sharpness of. (N, H, W)

    Returns:
        np.ndarray: The sharpness of the images. float, (N,)"""
    f = np.fft.fft2(imgs, axes=(1,2))
    f = np.real(f)
    fshift = np.fft.fftshift(f, axes=(1,2))
    AF = abs(fshift)
    thresh = AF.max(axis=(1,2)) / 2500
    th = (fshift > thresh[:,None,None]).sum(axis=(1,2))
    return th / (imgs.shape[1] * imgs.shape[2])


def get_sharpness(ebsd_data: namedtuple, use_torch: bool = False, batch_size: int = 8, lazy: bool = True, parallel: bool = True) -> np.ndarray:
    """Calculate the sharpness of a pattern object.
    This function can be used with either numpy for CPU or torch for GPU/MPS.
    If CPU is used, this can be parallelized if desired.
    It is also RAM safe by using lazy loading if desired.

    Args:
        ebsd_data (namedtuple): The pattern object to calculate the sharpness of.
        use_torch (bool): Whether to use torch or numpy.
        batch_size (int): The number of patterns to process at once.
        lazy (bool): Whether to use lazy loading.
        parallel (bool): Whether to parallelize the calculation.

    Returns:
        np.ndarray: The sharpness of the patterns."""
    if use_torch:
        return _get_sharpness_torch(ebsd_data, batch_size, lazy)
    else:
        return _get_sharpness_numpy(ebsd_data, batch_size, parallel, lazy)


# TODO: numpy and torch do not give the same results

if __name__ == "__main__":
    ang = "E:/GaN/GaN.ang"
    up2 = "E:/GaN/GaN.up2"
    ebsd_data = Data.EBSDData(up2, ang)

    t0 = time.time()
    sharpness0 = get_sharpness(ebsd_data, use_torch=True, batch_size=16, lazy=True, parallel=False).reshape(200, 200)
    print("Torch lazy:", time.time() - t0)

    t0 = time.time()
    sharpness2 = get_sharpness(ebsd_data, use_torch=False, batch_size=16, lazy=True, parallel=False).reshape(200, 200)
    print("Numpy lazy:", time.time() - t0)

    print(np.allclose(sharpness0, sharpness2, atol=1e-5, rtol=1e-5))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sharpness0)
    ax[0].set_title("Torch lazy")
    ax[1].imshow(sharpness2)
    ax[1].set_title("Numpy lazy")
    plt.show()

    # sharpness = sharpness.reshape(shape)
    # print(f"Saving sharpness to {dirname}")
    # io.imsave(os.path.join(dirname, filename.split(".")[0] + "_raw.tiff"), sharpness)
    # sharpness = np.around((sharpness - sharpness.min()) / (sharpness.max() - sharpness.min()) * 65535).astype('uint16')
    # io.imsave(os.path.join(dirname, filename.split(".")[0] + "_uint16.tiff"), sharpness)
    # print("Done")
