# Description: Utility functions for reading in EBSD data and processing patterns.
# Author: James Lamb

import os
import re
import struct
from collections import namedtuple

import numpy as np
from scipy import signal
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage import exposure, filters

import torch
import kornia

import ebsd_pattern as ep
import rotations

NUMERIC = r"[-+]?\d*\.\d+|\d+"


def convert_pc(PC: tuple | list | np.ndarray, N: tuple, delta: float, b: float = 1.0) -> tuple:
    """
    Converts the pattern center from EDAX/TSL standard to the EMsoft standard
    (xstar, ystar, zstar) -> (xpc, ypc, L)

    Args:
        PC (array-like): (xstar, ystar, zstar)
        N (array-like): detector dimensions before binning (Nx, Ny)
        delta: float, the detector pixel size
        b: float, some constant that is 1

    Returns:
        PC (tuple): The pattern center (xpc, ypc, L)"""
    xpc = np.around(N[0] * (PC[0] - 0.5), 4)
    ypc = np.around(N[0] * PC[1] - b * N[1] * 0.5, 4)
    L = np.around(N[0] * delta * PC[2], 4)
    return (xpc, ypc, L)


def read_up2(up2: str) -> namedtuple:
    """Read in patterns and a pattern center from an ang file and a pattern file.
    Only supports a up2 file using the EDAX/TSL convention.

    Args:
        up2 (str): Path to the pattern file.

    Returns:
        namedtuple: Pattern file object with fields patshape, filesize, nPatterns, and datafile.
                    patshape is a tuple of the pattern dimensions.
                    filesize is the size of the pattern file.
                    nPatterns is the number of patterns in the file.
                    datafile is the file object to read the patterns."""
    # Get patterns
    upFile = open(up2, "rb")
    chunk_size = 4
    tmp = upFile.read(chunk_size)
    FirstEntryUpFile = struct.unpack('i', tmp)[0]
    tmp = upFile.read(chunk_size)
    sz1 = struct.unpack('i', tmp)[0]
    tmp = upFile.read(chunk_size)
    sz2 = struct.unpack('i', tmp)[0]
    tmp = upFile.read(chunk_size)
    bitsPerPixel = struct.unpack('i', tmp)[0]
    sizeBytes = os.path.getsize(up2) - 16
    sizeString = str(round(sizeBytes / 1e6, 1)) + " MB"
    bytesPerPixel = 2
    nPatternsRecorded = int((sizeBytes/bytesPerPixel) / (sz1 * sz2))
    out = namedtuple("up2_file", ["patshape", "filesize", "nPatterns", "datafile"])
    out = out((sz1, sz2), sizeString, nPatternsRecorded, upFile)
    return out


def read_ang(path: str, Nxy: tuple, pixel_size: float = 10.0) -> namedtuple:
    """Reads in the pattern center from an ang file.
    Only supports EDAX/TSL.

    To print the data columns in the ang file, use the following:
    >>> ang_data = read_ang("path/to/ang/file.ang")
    >>> print(ang_data._fields)

    Args:
        ang (str): Path to the ang file.
        Nxy (tuple): The detector dimensions before binning. Used for converting the pattern center.
        pixel_size (float): The detector pixel size. Used for converting the pattern center.

    Returns:
        namedtuple: The data read in from the ang file with the following fields:
                    - quats: The quaternions.
                    - eulers: The Euler angles.
                    - shape: The shape of the data.
                    - pc: The pattern center.
                    - pidx: The index of the pattern in the pattern file.
                    - all data columns in the ang file (i.e. x, y, iq, ci, sem, phase_index, etc.)"""
    header_lines = 0
    with open(path, "r") as ang:
        for line in ang:
            if "x-star" in line:
                xstar = float(re.findall(NUMERIC, line)[0])
            elif "y-star" in line:
                ystar = float(re.findall(NUMERIC, line)[0])
            elif "z-star" in line:
                zstar = float(re.findall(NUMERIC, line)[0])
            elif "NROWS" in line:
                rows = int(re.findall(NUMERIC, line)[0])
            elif "NCOLS_ODD" in line:
                cols = int(re.findall(NUMERIC, line)[0])
            elif "COLUMN_HEADERS" in line:
                names = line.replace("\n", "").split(":")[1].strip().split(", ")
            elif "HEADER: End" in line:
                break
            header_lines += 1

    # Package the header data
    PC = convert_pc((xstar, ystar, zstar), Nxy, pixel_size)
    shape = (rows, cols)
    names.extend(["eulers", "quats", "shape", "pc", "pidx"])
    names = [name.replace(" ", "_").lower() for name in names if name.lower() not in ["phi1", "phi", "phi2"]]

    # Read in the data
    ang_data = np.genfromtxt(path, skip_header=header_lines)
    ang_data = ang_data.reshape(shape + (ang_data.shape[1],))
    euler = ang_data[..., 0:3]
    ang_data = ang_data[..., 3:]
    qu = rotations.eu2qu(euler)
    pidx = np.arange(np.prod(shape)).reshape(shape)
    ang_data = np.moveaxis(ang_data, 2, 0)

    # Package everything into a namedtuple
    ang_data = namedtuple("ang_file", names)(*ang_data, eulers=euler, quats=qu, shape=shape, pc=PC, pidx=pidx)
    return ang_data


def get_scan_data(up2: str, ang: str, Nxy: tuple, pixel_size: float = 10.0) -> tuple:
    """Reads in patterns and orientations from an ang file and a pattern file.
    Only supports EDAX/TSL.
    
    Args:
        up2 (str): Path to the pattern file.
        ang (str): Path to the ang file.
        Nxy (tuple): The detector dimensions before binning. Used for converting the pattern center.
        pixel_size (float): The detector pixel size. Used for converting the pattern center.

    Returns:
        np.ndarray: The patterns.
        namedtuple: The orientations. namedtuple with fields corresponding to the columns in the ang file + eulers, quats, shape, pc.
                    Sharpness replaces image quality (iq) if calculate_sharpness is True."""
    # Get the patterns
    pat_obj = read_up2(up2)

    # Get the ang data
    ang_data = read_ang(ang, ang_data.pc, Nxy, pixel_size)

    return pat_obj, ang_data


def get_patterns(pat_obj: namedtuple, idx: np.ndarray | list | tuple = None) -> tuple:
    """Read in patterns from a pattern file object.
    
    Args:
        pat_obj (namedtuple): Pattern file object.
        idx (np.ndarray | list | tuple): Indices of patterns to read in. If None, reads in all patterns.

    Returns:
        np.ndarray: Patterns."""
    # Handle inputs
    if idx is None:
        idx = range(pat_obj.nPatterns)
    elif isinstance(idx, np.ndarray):
        reshape = False
        if idx.ndim >= 2:
            reshape = True
            out_shape = idx.shape + pat_obj.patshape
            idx = idx.flatten()
    else:
        reshape = False

    # Read in the patterns
    pat_obj.datafile.seek(16)
    pats = np.zeros((len(idx), *pat_obj.patshape), dtype=np.uint16)
    for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
        pat = idx[i]
        pat_obj.datafile.seek(pat * pat_obj.patshape[0] * pat_obj.patshape[1] * 2 + 16)
        pats[i] = np.frombuffer(pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2), dtype=np.uint16).reshape(pat_obj.patshape)

    # Format and reshape the patterns
    pats = pats.astype(np.float32)
    mins = pats.min(axis=(1, 2))[:, None, None]
    maxs = pats.max(axis=(1, 2))[:, None, None]
    pats = (pats - mins) / (maxs - mins)
    if reshape:
        pats = pats.reshape(out_shape)

    return pats


def get_sharpness(imgs: np.ndarray) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.
    
    Args:
        imgs (np.ndarray): The images to calculate the sharpness of. (H, W) or (N, H, W) or (N0, N1, H, W)

    Returns:
        np.ndarray: The sharpness of the images. float, (N,) or (N0, N1) for example."""
    # Process inputs
    reshape = None
    if imgs.ndim == 2:
        imgs = imgs[None, ...]
    elif imgs.ndim == 3:
        reshape = imgs.shape[:1]
    elif imgs.ndim == 4:
        reshape = imgs.shape[:2]
        imgs = imgs.reshape(-1, *imgs.shape[2:])

    # Convert to torch tensor, set device, create output tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).to(device)
    shp = torch.zeros(imgs.shape[0], device=device)

    # Calculate sharpness
    for i in tqdm(range(imgs.shape[0]), desc='Calculating sharpness', unit='pats'):
        f = torch.fft.fft2(imgs[i])
        f = torch.real(f)
        f = torch.fft.fftshift(f)
        AF = torch.abs(f)
        M = AF.max()
        thresh = M / 2500
        th = (f > thresh).sum()
        shp[i] = th / (imgs.shape[2] * imgs.shape[3])

    # Convert to numpy and reshape if necessary
    shp = np.squeeze(shp.cpu().numpy())
    if reshape is not None:
        shp = shp.reshape(reshape)

    return shp


def process_patterns(imgs: np.ndarray, equalize: bool = True, dog_sigmas: tuple = None) -> np.ndarray:
    """Cleans patterns by equalizing the histogram and normalizing.
    
    Args:
        pats (np.ndarray): The patterns to clean. (N, H, W)
        equalize (bool): Whether to equalize the histogram.

    Returns:
        np.ndarray: The cleaned patterns. (N, H, W)"""
    # Process inputs
    reshape = None
    if imgs.ndim == 2:
        imgs = imgs[None, ...]
    elif imgs.ndim == 3:
        reshape = imgs.shape[:1]
    elif imgs.ndim == 4:
        reshape = imgs.shape[:2]
        imgs = imgs.reshape(-1, *imgs.shape[2:])
    imgs = imgs.astype(np.float32).reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])
    if dog_sigmas is None:
        bandpass = False
    else:    
        bandpass = True

    # Convert to torch tensor, set device, create output tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs = torch.tensor(imgs, dtype=torch.float32).to(device)

    # Subtract the background
    # avg = imgs.mean(dim=(0), keepdim=True)
    # background = kornia.filters.gaussian_blur2d(avg, (3, 3), (10.0, 10.0))
    # imgs = imgs - background
    
    # Create processing functions
    def make_odd(x):
        return  x if x % 2 == 1 else x + 1
    def NONE(img):
        return img
    def equalize_func(img):
        equalized = kornia.enhance.equalize_clahe(img)
        out = kornia.enhance.normalize_min_max(equalized, 0.0, 1.0)
        return out
    def bandpass_func(img):
        kl = make_odd(int(dog_sigmas[0] * 3))
        kh = make_odd(int(dog_sigmas[1] * 3))
        low_pass = kornia.filters.gaussian_blur2d(img, (kl, kl), (dog_sigmas[0], dog_sigmas[0]))
        high_pass = kornia.filters.gaussian_blur2d(img, (kh, kh), (dog_sigmas[1], dog_sigmas[1]))
        out = kornia.enhance.normalize_min_max(low_pass - high_pass, 0.0, 1.0)
        return out
    def all_func(img):
        return equalize_func(bandpass_func(img))

    # Reshape into batches and process patterns
    imgs = imgs.reshape(-1, 4, 1, imgs.shape[2], imgs.shape[3])
    func = all_func if equalize and bandpass else equalize_func if equalize else bandpass_func if bandpass else NONE
    for i in tqdm(range(imgs.shape[0]), desc='Processing patterns', unit='pats'):
        imgs[i] = func(kornia.enhance.normalize_min_max(imgs[i], 0.0, 1.0))

    # Normalize
    out = imgs.cpu().numpy()
    out = out.reshape(-1, out.shape[3], out.shape[4])
    if reshape is not None:
        out = out.reshape(reshape + out.shape[1:])

    # Clear memory
    del imgs, background, avg
    torch.cuda.empty_cache()

    return out


def get_index(point: tuple, size: int, ang_data: namedtuple) -> np.ndarray:
    """Get the indices of the patterns that reside within a region of interest.
    
    Args:
        point (tuple): The center of the region of interest.
        size (int): The size of the region of interest.
        ang_data (namedtuple): The ang data.

    Returns:
        np.ndarray: The indices of the patterns within the region of interest."""
    # Get the indices of the patterns in the region of interest
    x, y = point
    x0, x1 = min(x - size // 2, 0), max(x + size // 2, ang_data.shape[0])
    y0, y1 = min(y - size // 2, 0), max(y + size // 2, ang_data.shape[1])
    if x1 - x0 < size:
        print(" -- get_index warning: the region of interest is too large in the x-direction given the center point.")
    if y1 - y0 < size:
        print(" -- get_index warning: the region of interest is too large in the y-direction given the center point.")
    idx = ang_data.pidx[x0:x1, y0:y1]
    return idx


def test_bandpass(img, save_dir="./"):
    """Run bandpass filtering (using difference of gaussians) on an image.
    Do it for a range of lower and upper sigma values.
    Do vertical and horizontal stacking of the results to create on image (vertical axis is the high pass, horizontal is the low pass).
    For each image, also compute the fft and create the same composite image.
    
    Args:
        img (np.ndarray): The image to filter."""
    # Process inputs
    c = np.array(img.shape) // 2
    slc = (slice(c[0] - 64, c[0] + 64), slice(c[1] - 64, c[1] + 64))
    img = img[slc]
    low_sigmas = np.arange(0.25, 2.0, 0.25)  # high pass
    high_sigmas = np.arange(4.0, 11.0, 1.0)  # low pass
    composite = np.zeros((len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1]))
    composite_eq = np.zeros((len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1]))
    composite_xcf = np.zeros((len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1]))
    composite_xcf_eq = np.zeros((len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1]))
    window = (signal.windows.tukey(img.shape[0], alpha=0.4)[:, None] * signal.windows.tukey(img.shape[1], alpha=0.4)[None, :])
    sigmas = np.zeros((len(low_sigmas), len(high_sigmas), 2))
    for i, l in enumerate(low_sigmas):
        for j, h in enumerate(high_sigmas):
            # Apply the bandpass filter
            image = filters.difference_of_gaussians(img, low_sigma=l, high_sigma=h)
            image = (image - image.min()) / (image.max() - image.min())
            # Apply a tukey hann window
            image = image * window
            composite[i*img.shape[0]:(i+1)*img.shape[0], j*img.shape[1]:(j+1)*img.shape[1]] = image
            # Compute the cross-correlation
            xcf = signal.correlate2d(image, image, mode='same')
            composite_xcf[i*img.shape[0]:(i+1)*img.shape[0], j*img.shape[1]:(j+1)*img.shape[1]] = xcf
            # Equalize the histogram
            image = exposure.equalize_adapthist(image)
            composite_eq[i*img.shape[0]:(i+1)*img.shape[0], j*img.shape[1]:(j+1)*img.shape[1]] = image
            # Compute the cross-correlation
            xcf = signal.correlate2d(image, image, mode='same')
            composite_xcf_eq[i*img.shape[0]:(i+1)*img.shape[0], j*img.shape[1]:(j+1)*img.shape[1]] = xcf
            sigmas[i, j] = (l, h)
    
    # Now plot the two composite images using matplotlib with the sigmas as ticklabels
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].imshow(composite, cmap='gray')
    ax[0, 0].set_xticks(np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2)
    ax[0, 0].set_xticklabels(high_sigmas)
    ax[0, 0].set_yticks(np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2)
    ax[0, 0].set_yticklabels(low_sigmas)
    ax[0, 0].set_title("Difference of Gaussians")
    ax[0, 0].set_xlabel("High sigma")
    ax[0, 0].set_ylabel("Low sigma")

    ax[0, 1].imshow(composite_eq, cmap='gray')
    ax[0, 1].set_xticks(np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2)
    ax[0, 1].set_xticklabels(high_sigmas)
    ax[0, 1].set_yticks(np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2)
    ax[0, 1].set_yticklabels(low_sigmas)
    ax[0, 1].set_title("Difference of Gaussians + Equalization")
    ax[0, 1].set_xlabel("High sigma")
    ax[0, 1].set_ylabel("Low sigma")

    ax[1, 0].imshow(composite_xcf, cmap='gray')
    ax[1, 0].set_xticks(np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2)
    ax[1, 0].set_xticklabels(high_sigmas)
    ax[1, 0].set_yticks(np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2)
    ax[1, 0].set_yticklabels(low_sigmas)
    ax[1, 0].set_title("Cross-correlation of DoG")
    ax[1, 0].set_xlabel("High sigma")
    ax[1, 0].set_ylabel("Low sigma")

    ax[1, 1].imshow(composite_xcf_eq, cmap='gray')
    ax[1, 1].set_xticks(np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2)
    ax[1, 1].set_xticklabels(high_sigmas)
    ax[1, 1].set_yticks(np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2)
    ax[1, 1].set_yticklabels(low_sigmas)
    ax[1, 1].set_title("Cross-correlation of DoG + Equalization")
    ax[1, 1].set_xlabel("High sigma")
    ax[1, 1].set_ylabel("Low sigma")

    fig.tight_layout()
    fig.savefig(save_dir + "bandpass_test.png")
    plt.close(fig)
    return composite, composite_xcf


def view(*imgs, cmap='gray', titles=None, save_dir=None):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(imgs):
        ax = axes[i]
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def view_tensor_images(e, cmap="jet", tensor_type="strain", xy=None, save_dir=None, save_name=""):
    """View individual tensor components of a grid of tensors (such as the strain tensor from HREBSD).
    
    Args:
        e (np.ndarray): The tensor components. (H, W, 3, 3)
        cmap (str): The colormap to use.
        tensor_type (str): The type of tensor. Options are "strain", "rotation", and "deformation".
        xy (tuple): The point to plot on the tensor components.
        save_dir (str): The directory to save the image.
        save_name (str): The name of the image. The filename will be save_name + tensor_type + ".png"."""
    if tensor_type == "strain":
        var = r"$\epsilon$"
    elif tensor_type == "rotation":
        var = r"$\omega$"
    elif tensor_type == "deformation":
        var = r"$F$"
    fig, ax = plt.subplots(3, 3, figsize=(12.2, 12))
    plt.subplots_adjust(wspace=0.35, hspace=0.01, left=0.01, right=0.93, top=0.99, bottom=0.01)
    for i in range(3):
        for j in range(3):
            _0 = ax[i, j].imshow(e[..., i, j], cmap=cmap, vmin=-abs(e[..., i, j]).max(), vmax=abs(e[..., i, j]).max())
            ax[i, j].axis('off')
            ax[i, j].set_title(var + r"$_{%i%i}$" % (i+1, j+1))
            if xy is not None:
                ax[i, j].plot(xy[1], xy[0], 'kx', markersize=10)
            loc = ax[i, j].get_position()
            cax = fig.add_axes([loc.x1 + 0.01, loc.y0, 0.01, loc.height])
            cbar = fig.colorbar(_0, cax=cax, orientation='vertical')
            cbar.formatter.set_powerlimits((-1, 1))
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{save_name}{tensor_type}.png"), dpi=300)
        plt.close(fig)
    plt.show()
