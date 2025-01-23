# Description: Utility functions for reading in EBSD data and processing patterns.
# Author: James Lamb

import os
import re
import struct
import itertools
from collections import namedtuple

import numpy as np
from scipy import signal, ndimage
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage import exposure, filters

import torch
import kornia

import segment
import rotations

NUMERIC = r"[-+]?\d*\.\d+|\d+"


def convert_pc(
    PC: tuple | list | np.ndarray, patshape: tuple | list | np.ndarray
) -> tuple:
    """
    Converts the pattern center from EDAX/TSL standard to the EMsoft standard
    (xstar, ystar, zstar) -> (xpc, ypc, L)

    Args:
        PC (array-like): (xstar, ystar, zstar) --OR-- (xpc, ypc, L). If the latter is given, the L parameter needs to be in units of pixels.
        N (array-like): detector dimensions after binning, aka the pattern dimensions (Nx, Ny)
        delta (float): the raw detector pixel size before binning
        b (float): the binning factor

    Returns:
        PC (tuple): The pattern center (xpc, ypc, L) all in units of pixels.
        --OR--
        PC (tuple): The pattern center (xstar, ystar, zstar)."""
    xpc = PC[0] * patshape[1]
    ypc = PC[1] * patshape[0]
    DD = PC[2] * patshape[0]
    return (xpc, ypc, DD)


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
    FirstEntryUpFile = struct.unpack("i", tmp)[0]
    tmp = upFile.read(chunk_size)
    sz1 = struct.unpack("i", tmp)[0]
    tmp = upFile.read(chunk_size)
    sz2 = struct.unpack("i", tmp)[0]
    tmp = upFile.read(chunk_size)
    bitsPerPixel = struct.unpack("i", tmp)[0]
    sizeBytes = os.path.getsize(up2) - 16
    sizeString = str(round(sizeBytes / 1e6, 1)) + " MB"
    bytesPerPixel = 2
    nPatternsRecorded = int((sizeBytes / bytesPerPixel) / (sz1 * sz2))
    out = namedtuple("up2_file", ["patshape", "filesize", "nPatterns", "datafile"])
    out = out((sz1, sz2), sizeString, nPatternsRecorded, upFile)
    return out


def read_ang(path: str, patshape: tuple | list | np.ndarray = None, segment_grain_threshold: float = None) -> namedtuple:
    """Reads in the pattern center from an ang file.
    Only supports EDAX/TSL.

    To print the data columns in the ang file, use the following:
    >>> ang_data = read_ang("path/to/ang/file.ang")
    >>> print(ang_data._fields)

    Args:
        ang (str): Path to the ang file.
        patshape (tuple): The shape of the patterns. If None, the pattern center will be
                          (xstar, ystar, zstar) and not (xpc, ypc, L).
        segment_grain_threshold (bool): Grain boundary threshold for segmenting grains.
                                        If None (default), grains will not be segmented.

    Returns:
        namedtuple: The data read in from the ang file with the following fields:
                    - quats: The quaternions.
                    - eulers: The Euler angles.
                    - shape: The shape of the data.
                    - pc: The pattern center.
                    - pidx: The index of the pattern in the pattern file.
                    - ids: The grain IDs. Will be all ones if segment_grains is False.
                    - all data columns in the ang file
                      (i.e. x, y, iq, ci, sem, phase_index, etc.)
    """
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
            elif "XSTEP" in line:
                step_size = float(re.findall(NUMERIC, line)[0])
            elif "COLUMN_HEADERS" in line:
                names = line.replace("\n", "").split(":")[1].strip().split(", ")
            elif "HEADER: End" in line:
                break
            header_lines += 1

    # Package the header data
    if patshape is not None:
        PC = convert_pc((xstar, ystar, zstar), patshape)
    else:
        PC = (xstar, ystar, zstar)
    shape = (rows, cols)
    names.extend(["eulers", "quats", "shape", "pc", "step_size", "pidx"])
    if segment_grain_threshold is not None:
        names.extend(["ids", "kam"])
    names = [
        name.replace(" ", "_").lower()
        for name in names
        if name.lower() not in ["phi1", "phi", "phi2"]
    ]

    # Read in the data
    ang_data = np.genfromtxt(path, skip_header=header_lines)
    ang_data = ang_data.reshape(shape + (ang_data.shape[1],))
    euler = ang_data[..., 0:3]
    ang_data = ang_data[..., 3:]
    qu = rotations.eu2qu(euler)
    pidx = np.arange(np.prod(shape)).reshape(shape)
    if segment_grain_threshold is not None:
        ids, kam = segment.segment_grains(qu, segment_grain_threshold)
        args = (euler, qu, shape, PC, step_size, pidx, ids, kam)
    else:
        args = (euler, qu, shape, PC, step_size, pidx)
    ang_data = np.moveaxis(ang_data, 2, 0)

    # Package everything into a namedtuple
    out = namedtuple("ang_file", names)(*ang_data, *args)
    return out


def get_scan_data(up2: str, ang: str) -> tuple:
    """Reads in patterns and orientations from an ang file and a pattern file.
    Only supports EDAX/TSL.

    Args:
        up2 (str): Path to the pattern file.
        ang (str): Path to the ang file.

    Returns:
        np.ndarray: The patterns.
        namedtuple: The orientations. namedtuple with fields corresponding to the columns in the ang file + eulers, quats, shape, pc.
    """
    # Get the patterns
    pat_obj = read_up2(up2)

    # Get the ang data
    ang_data = read_ang(ang, pat_obj.patshape)

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
        reshape = False
    else:
        idx = np.asarray(idx)
        reshape = False
        if idx.ndim >= 2:
            reshape = True
            out_shape = idx.shape + pat_obj.patshape
            idx = idx.flatten()

    # Read in the patterns
    start_byte = np.int64(16)
    pattern_bytes = np.int64(pat_obj.patshape[0] * pat_obj.patshape[1] * 2)
    pats = np.zeros((len(idx), *pat_obj.patshape), dtype=np.uint16)
    # for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
    for i in range(len(idx)):
        pat = np.int64(idx[i])
        seek_pos = np.int64(start_byte + pat * pattern_bytes)
        pat_obj.datafile.seek(seek_pos)
        pats[i] = np.frombuffer(
            pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2),
            dtype=np.uint16,
        ).reshape(pat_obj.patshape)

    # Reshape the patterns
    pats = np.squeeze(pats)
    if reshape:
        pats = pats.reshape(out_shape)

    return pats


def get_pattern(pat_obj: namedtuple, idx: int = None) -> tuple:
    """Read in patterns from a pattern file object.

    Args:
        pat_obj (namedtuple): Pattern file object.
        idx (int): Indice of pattern to read in.

    Returns:
        np.ndarray: Patterns."""
    # Read in the patterns
    start_byte = np.int64(16)
    pattern_bytes = np.int64(pat_obj.patshape[0] * pat_obj.patshape[1] * 2)
    # for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
    seek_pos = np.int64(start_byte + np.int64(idx) * pattern_bytes)
    pat_obj.datafile.seek(seek_pos)
    pat = np.frombuffer(
        pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2),
        dtype=np.uint16,
    ).reshape(pat_obj.patshape)
    return pat


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).to(device)
    shp = torch.zeros(imgs.shape[0], device=device)

    # Calculate sharpness
    for i in tqdm(range(imgs.shape[0]), desc="Calculating sharpness", unit="pats"):
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


def process_patterns_gpu(
    imgs: np.ndarray,
    sigma: float = 0.0,
    equalize: bool = True,
    truncate: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    """Cleans patterns by equalizing the histogram and normalizing.

    Args:
        pats (np.ndarray): The patterns to clean. (N, H, W)
        equalize (bool): Whether to equalize the histogram.
        high_pass (bool): Whether to apply a high-pass filter.
        truncate (bool): Whether to truncate the patterns.
        batch_size (int): The batch size for processing the patterns.

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
    imgs = imgs.astype(np.float32).reshape(
        imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]
    )
    median = np.percentile(imgs, 50)
    low_percentile = np.percentile(imgs, 1)
    high_percentile = np.percentile(imgs, 99)
    # background = ndimage.gaussian_filter(imgs.mean(axis=0), imgs.shape[-1] / 10)
    # imgs = imgs - background
    # imgs = (imgs - imgs.min(axis=(1,2))[:,None,None]) / (imgs.max(axis=(1,2)) - imgs.min(axis=(1,2)))[:,None,None]

    # Convert to torch tensor, set device, create output tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = torch.tensor(imgs, dtype=torch.float32).to("cpu")

    # Create processing functions
    def get_kernel(sigma):
        k = sigma if sigma % 2 == 1 else sigma + 1
        return (int(k), int(k)), (float(sigma), float(sigma))

    # Process the patterns, using batches
    bk, bs = get_kernel(int(imgs.shape[-1] / 10))
    imgs_batched = list(torch.split(imgs, batch_size, dim=0))
    for i in tqdm(range(len(imgs_batched)), desc="Processing patterns", unit="batches"):
        imgs = imgs_batched[i].to(device)
        if truncate:
            imgs[imgs < low_percentile] = median
            imgs[imgs > high_percentile] = median
        imgs = kornia.enhance.normalize_min_max(imgs, 0.0, 1.0)
        background = kornia.filters.gaussian_blur2d(imgs, bk, bs)
        imgs = imgs - background
        imgs = kornia.enhance.normalize_min_max(imgs, 0.0, 1.0)
        if equalize:
            imgs = kornia.enhance.equalize_clahe(imgs)
            imgs = kornia.enhance.normalize_min_max(imgs, 0.0, 1.0)
        if sigma > 0.0:
            k, s = get_kernel(sigma)
            imgs = kornia.filters.gaussian_blur2d(imgs, k, s)
            imgs = kornia.enhance.normalize_min_max(imgs, 0.0, 1.0)

        imgs_batched[i] = imgs.to("cpu")
    if len(imgs_batched) > 1:
        imgs = torch.cat(imgs_batched, dim=0)
    else:
        imgs = imgs_batched[0]

    # Normalize
    out = imgs.cpu().numpy()
    out = out.reshape(-1, out.shape[2], out.shape[3])
    if reshape is not None:
        out = out.reshape(reshape + out.shape[1:])

    # Clear memory
    del imgs
    torch.cuda.empty_cache()

    return out


def process_pattern(
    img: np.ndarray, low_pass_sigma: float = 2.5, high_pass_sigma: float = 101, truncate_std_scale: float = 3.0
) -> np.ndarray:
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
    low_percentile = np.percentile(img, 0.1)
    high_percentile = np.percentile(img, 99.9)
    img[img < low_percentile] = median
    img[img > high_percentile] = median

    # Low pass filter
    img = (img - img.min()) / (img.max() - img.min())
    img = ndimage.gaussian_filter(img, low_pass_sigma)

    # High pass filter
    background = ndimage.gaussian_filter(img, high_pass_sigma)
    img = img - background
    img = (img - img.min()) / (img.max() - img.min())

    # Truncate step
    mean, std = img.mean(), img.std()
    img = np.clip(img, mean - truncate_std_scale * std, mean + truncate_std_scale * std)

    return img


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
    x0, x1 = max(x - size // 2, 0), min(x + size // 2, ang_data.shape[0])
    y0, y1 = max(y - size // 2, 0), min(y + size // 2, ang_data.shape[1])
    if x1 - x0 < size:
        print(
            " -- get_index warning: the region of interest is too large in the x-direction given the center point."
        )
    if y1 - y0 < size:
        print(
            " -- get_index warning: the region of interest is too large in the y-direction given the center point."
        )
    idx = ang_data.pidx[x0:x1, y0:y1]
    return idx


def get_stiffness_tensor(*C, structure) -> np.ndarray:
    """Convert format elastic constants into the full stiffness tensor.
    Supports cubic and hexagonal crystal structures.

    Args:
        C (tuple): The elastic constants.
                   For cubic, C11, C12, C44.
                   For hexagonal, C11, C12, C13, C33, C44.
        structure (str): The crystal structure. Options are "cubic" and "hexagonal"."""
    C = np.array(C)
    if structure == "cubic":
        if C.shape[0] != 3:
            raise ValueError(
                "Cubic crystal structure requires 3 elastic constants in the following order: C11, C12, C44."
            )
        C = np.array(
            [
                [C[0], C[1], C[1], 0, 0, 0],
                [C[1], C[0], C[1], 0, 0, 0],
                [C[1], C[1], C[0], 0, 0, 0],
                [0, 0, 0, C[2], 0, 0],
                [0, 0, 0, 0, C[2], 0],
                [0, 0, 0, 0, 0, C[2]],
            ]
        )
    elif structure == "hexagonal":
        if C.shape[0] != 5:
            raise ValueError(
                "Hexagonal crystal structure requires 5 elastic constants in the following order: C11, C12, C13, C33, C44."
            )
        C66 = 0.5 * (C[0] - C[1])
        C = np.array(
            [
                [C[0], C[1], C[2], 0, 0, 0],
                [C[1], C[0], C[2], 0, 0, 0],
                [C[2], C[2], C[3], 0, 0, 0],
                [0, 0, 0, C[4], 0, 0],
                [0, 0, 0, 0, C[4], 0],
                [0, 0, 0, 0, 0, C66],
            ]
        )
    else:
        raise ValueError(
            "Unsupported crystal structure. Options are 'cubic' and 'hexagonal'."
        )
    return C


def rotate_stiffness_to_sample_frame(C: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Rotate the stiffness tensor to the sample frame.

    Args:
        C (np.ndarray): The stiffness tensor.
        quats (np.ndarray): The quaternions.

    Returns:
        np.ndarray: The rotated stiffness tensor."""
    if quats.ndim == 1:
        quats = quats[None, :]
        out_shape = (6, 6)
    else:
        out_shape = quats.shape[:-1] + (6, 6)
        quats = quats.reshape(-1, 4)
    # Rotate the stiffness tensor to the sample frame
    C_rot = np.zeros((quats.shape[0], 6, 6))
    for i in range(quats.shape[0]):
        R = rotations.qu2om(quats[i]).T
        C_rot[i] = rotate_elastic_constants(C, R)
    C_rot = C_rot.reshape(out_shape)
    return C_rot


def test_bandpass(img, save_dir="./", window_size=128):
    """Run bandpass filtering (using difference of gaussians) on an image.
    Do it for a range of lower and upper sigma values.
    Do vertical and horizontal stacking of the results to create on image (vertical axis is the high pass, horizontal is the low pass).
    For each image, also compute the fft and create the same composite image.

    Args:
        img (np.ndarray): The image to filter."""
    # Process inputs
    c = np.array(img.shape) // 2
    slc = (
        slice(c[0] - window_size // 2, c[0] + window_size // 2),
        slice(c[1] - window_size // 2, c[1] + window_size // 2),
    )
    img = img[slc]
    low_sigmas = np.arange(0.25, 2.0, 0.25)  # high pass
    high_sigmas = np.arange(5.0, 40.0, 5.0)  # low pass
    composite = np.zeros(
        (len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1])
    )
    composite_eq = np.zeros(
        (len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1])
    )
    composite_xcf = np.zeros(
        (len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1])
    )
    composite_xcf_eq = np.zeros(
        (len(low_sigmas) * img.shape[0], len(high_sigmas) * img.shape[1])
    )
    window = (
        signal.windows.tukey(img.shape[0], alpha=0.4)[:, None]
        * signal.windows.tukey(img.shape[1], alpha=0.4)[None, :]
    )
    sigmas = np.zeros((len(low_sigmas), len(high_sigmas), 2))
    for i, l in enumerate(low_sigmas):
        for j, h in enumerate(high_sigmas):
            # Apply the bandpass filter
            image = filters.difference_of_gaussians(img, low_sigma=l, high_sigma=h)
            image = (image - image.min()) / (image.max() - image.min())
            # Apply a tukey hann window
            image = image * window
            composite[
                i * img.shape[0] : (i + 1) * img.shape[0],
                j * img.shape[1] : (j + 1) * img.shape[1],
            ] = image
            # Compute the cross-correlation
            xcf = signal.fftconvolve(image, image[::-1, ::-1], mode="same").real
            composite_xcf[
                i * img.shape[0] : (i + 1) * img.shape[0],
                j * img.shape[1] : (j + 1) * img.shape[1],
            ] = xcf
            # Equalize the histogram
            image = exposure.equalize_adapthist(image)
            composite_eq[
                i * img.shape[0] : (i + 1) * img.shape[0],
                j * img.shape[1] : (j + 1) * img.shape[1],
            ] = image
            # Compute the cross-correlation
            xcf = signal.fftconvolve(image, image[::-1, ::-1], mode="same").real
            composite_xcf_eq[
                i * img.shape[0] : (i + 1) * img.shape[0],
                j * img.shape[1] : (j + 1) * img.shape[1],
            ] = xcf
            sigmas[i, j] = (l, h)

    # Now plot the two composite images using matplotlib with the sigmas as ticklabels
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].imshow(composite, cmap="gray")
    ax[0, 0].set_xticks(
        np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2
    )
    ax[0, 0].set_xticklabels(high_sigmas)
    ax[0, 0].set_yticks(
        np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2
    )
    ax[0, 0].set_yticklabels(low_sigmas)
    ax[0, 0].set_title("Difference of Gaussians")
    ax[0, 0].set_xlabel("High sigma")
    ax[0, 0].set_ylabel("Low sigma")

    ax[0, 1].imshow(composite_eq, cmap="gray")
    ax[0, 1].set_xticks(
        np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2
    )
    ax[0, 1].set_xticklabels(high_sigmas)
    ax[0, 1].set_yticks(
        np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2
    )
    ax[0, 1].set_yticklabels(low_sigmas)
    ax[0, 1].set_title("Difference of Gaussians + Equalization")
    ax[0, 1].set_xlabel("High sigma")
    ax[0, 1].set_ylabel("Low sigma")

    ax[1, 0].imshow(composite_xcf, cmap="gray")
    ax[1, 0].set_xticks(
        np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2
    )
    ax[1, 0].set_xticklabels(high_sigmas)
    ax[1, 0].set_yticks(
        np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2
    )
    ax[1, 0].set_yticklabels(low_sigmas)
    ax[1, 0].set_title("Cross-correlation of DoG")
    ax[1, 0].set_xlabel("High sigma")
    ax[1, 0].set_ylabel("Low sigma")

    ax[1, 1].imshow(composite_xcf_eq, cmap="gray")
    ax[1, 1].set_xticks(
        np.arange(0, composite.shape[1], img.shape[1]) + img.shape[1] // 2
    )
    ax[1, 1].set_xticklabels(high_sigmas)
    ax[1, 1].set_yticks(
        np.arange(0, composite.shape[0], img.shape[0]) + img.shape[0] // 2
    )
    ax[1, 1].set_yticklabels(low_sigmas)
    ax[1, 1].set_title("Cross-correlation of DoG + Equalization")
    ax[1, 1].set_xlabel("High sigma")
    ax[1, 1].set_ylabel("Low sigma")

    fig.tight_layout()
    fig.savefig(save_dir + "bandpass_test.png")
    plt.close(fig)


def view(*imgs, cmap="gray", titles=None, save_dir=None):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(imgs):
        ax = axes[i]
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def view_tensor_images(
    e,
    cmap="jet",
    tensor_type="strain",
    xy=None,
    save_dir=None,
    save_name="",
    show="all",
    clip="global",
):
    """View individual tensor components of a grid of tensors (such as the strain tensor from HREBSD).

    Args:
        e (np.ndarray): The tensor components. (H, W, 3, 3)
        cmap (str): The colormap to use.
        tensor_type (str): The type of tensor. Options are "strain", "rotation", and "deformation".
        xy (tuple): The point to plot on the tensor components.
        save_dir (str): The directory to save the image.
        save_name (str): The name of the image. The filename will be save_name + tensor_type + ".png".
        show (str): The tensor components to show. Options are "all", "diag", "upper", or "lower".
        clip (str): The clipping method. Options are "global" or "local". If "global", the min and max values are taken from the entire tensor.
                    If "local", the min and max values are taken from each tensor component.
    """
    # Process tensor type
    if tensor_type == "strain":
        var = r"$\epsilon$"
    elif tensor_type == "rotation":
        var = r"$\omega$"
    elif tensor_type == "deformation":
        var = r"$F$"
    elif tensor_type == "stress":
        var = r"$\sigma$"
    elif tensor_type == "homography":
        var = r"$h$"
    # Process show
    if show == "all":
        bad = []
    elif show == "diag":
        bad = ["01", "02", "10", "12", "20", "21"]
    elif show == "upper":
        bad = ["10", "20", "21"]
    elif show == "lower":
        bad = ["01", "02", "12"]
    if clip == "global":
        vmin = np.percentile(e, 1)
        vmax = np.percentile(e, 99)
    fig, ax = plt.subplots(3, 3, figsize=(12.2, 12))
    plt.subplots_adjust(
        wspace=0.35, hspace=0.01, left=0.01, right=0.93, top=0.99, bottom=0.01
    )
    for i in range(3):
        for j in range(3):
            if "%i%i" % (i, j) in bad:
                # Turn off the axis
                ax[i, j].axis("off")
                continue
            elif tensor_type == "homography":
                if i == 2 and j == 2:
                    ax[i, j].axis("off")
                    continue
            if clip == "local":
                vmin = np.percentile(e[..., i, j], 0.1)
                vmax = np.percentile(e[..., i, j], 99.9)
            if tensor_type == "homography":
                _0 = ax[i, j].imshow(e[..., 3 * i + j], cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                _0 = ax[i, j].imshow(e[..., i, j], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[i, j].axis("off")
            ax[i, j].set_title(var + r"$_{%i%i}$" % (i + 1, j + 1), fontsize=20)
            if xy is not None:
                ax[i, j].plot(xy[1], xy[0], "kx", markersize=10)
            loc = ax[i, j].get_position()
            cax = fig.add_axes([loc.x1 + 0.01, loc.y0, 0.01, loc.height])
            cbar = fig.colorbar(_0, cax=cax, orientation="vertical")
            cbar.formatter.set_powerlimits((-1, 1))
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, f"{save_name}_{tensor_type}.png"),
            dpi=300,
            transparent=True,
        )
        plt.close(fig)
    plt.show()


def shade_ipf(ipf, greyscale):
    # Process greyscale
    greyscale = greyscale.astype(np.float32)
    greyscale = (
        (greyscale - greyscale.min()) / (greyscale.max() - greyscale.min())
    ).reshape(greyscale.shape + (1,))

    # Shade the IPF
    ipf = ipf.astype(np.float32)
    ipf = ipf * greyscale
    return ipf


def make_video(folder, save_path):
    import cv2
    import os

    prefix = "img"
    ext = ".png"
    fps = 24

    fourcc = cv2.VideoWriter_fourcc(*"h264")
    images = sorted(
        [
            img
            for img in os.listdir(folder)
            if img.endswith(ext) and img[: len(prefix)] == prefix
        ],
        key=lambda x: int(x.split(".")[0].replace(prefix, "")),
    )
    frame = cv2.imread(os.path.join(folder, images[0]))
    h, w, l = frame.shape
    video = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for i, image in enumerate(images):
        video.write(cv2.imread(os.path.join(folder, image)))
    cv2.destroyAllWindows()
    video.release()


def standardize_axis(ax, **kwargs):
    kwargs["labelsize"] = kwargs.get("labelsize", 20)
    kwargs["labelcolor"] = kwargs.get("labelcolor", "k")
    kwargs["direction"] = kwargs.get("direction", "in")
    kwargs["top"] = kwargs.get("top", True)
    kwargs["right"] = kwargs.get("right", True)
    ax.tick_params(axis="both", which="both", **kwargs)
    ax.grid(alpha=0.3, which="major")
    ax.grid(alpha=0.1, which="minor")


def make_legend(ax, **kwargs):
    # kwargs["bbox_to_anchor"] = kwargs.get("bbox_to_anchor", (1.03, 1.05))
    # kwargs["loc"] = kwargs.get("loc", "upper right")
    kwargs["fontsize"] = kwargs.get("fontsize", 18)
    kwargs["shadow"] = kwargs.get("shadow", True)
    kwargs["framealpha"] = kwargs.get("framealpha", 1)
    kwargs["fancybox"] = kwargs.get("fancybox", False)
    ax.legend(**kwargs)


### Elastic constants conversion functions ###
# Taken from: https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py

Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def rotate_elastic_constants(C, A, tol=1e-6):
    """
    Return rotated elastic moduli for a general crystal given the elastic
    constant in Voigt notation.

    Parameters
    ----------
    C : array_like
        6x6 matrix of elastic constants (Voigt notation).
    A : array_like
        3x3 rotation matrix.

    Returns
    -------
    C : array
        6x6 matrix of rotated elastic constants (Voigt notation).
    """

    A = np.asarray(A)

    # Is this a rotation matrix?
    if np.sometrue(np.abs(np.dot(np.array(A), np.transpose(np.array(A))) -
                          np.eye(3, dtype=float)) > tol):
        raise RuntimeError('Matrix *A* does not describe a rotation.')

    # Rotate
    return full_3x3x3x3_to_Voigt_6x6(np.einsum('ia,jb,kc,ld,abcd->ijkl',
                                               A, A, A, A,
                                               Voigt_6x6_to_full_3x3x3x3(C)))


def full_3x3_to_Voigt_6_index(i, j):
    if i == j:
        return i
    return 6-i-j


def Voigt_6x6_to_full_3x3x3x3(C):
    """
    Convert from the Voigt representation of the stiffness matrix to the full
    3x3x3x3 representation.

    Parameters
    ----------
    C : array_like
        6x6 stiffness matrix (Voigt notation).

    Returns
    -------
    C : array_like
        3x3x3x3 stiffness matrix.
    """

    C = np.asarray(C)
    C_out = np.zeros((3,3,3,3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out


def full_3x3x3x3_to_Voigt_6x6(C, tol=1e-3, check_symmetry=True):
    """
    Convert from the full 3x3x3x3 representation of the stiffness matrix
    to the representation in Voigt notation. Checks symmetry in that process.
    """

    C = np.asarray(C)
    Voigt = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i,j] = C[k,l,m,n]
            """
            print('---')
            print("k,l,m,n", C[k,l,m,n])
            print("m,n,k,l", C[m,n,k,l])
            print("l,k,m,n", C[l,k,m,n])
            print("k,l,n,m", C[k,l,n,m])
            print("m,n,l,k", C[m,n,l,k])
            print("n,m,k,l", C[n,m,k,l])
            print("l,k,n,m", C[l,k,n,m])
            print("n,m,l,k", C[n,m,l,k])
            print('---')
            """
            if check_symmetry:
                assert abs(Voigt[i,j]-C[m,n,k,l]) < tol, \
                    '1 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], m, n, k, l, C[m,n,k,l])
                assert abs(Voigt[i,j]-C[l,k,m,n]) < tol, \
                    '2 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], l, k, m, n, C[l,k,m,n])
                assert abs(Voigt[i,j]-C[k,l,n,m]) < tol, \
                    '3 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], k, l, n, m, C[k,l,n,m])
                assert abs(Voigt[i,j]-C[m,n,l,k]) < tol, \
                    '4 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], m, n, l, k, C[m,n,l,k])
                assert abs(Voigt[i,j]-C[n,m,k,l]) < tol, \
                    '5 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], n, m, k, l, C[n,m,k,l])
                assert abs(Voigt[i,j]-C[l,k,n,m]) < tol, \
                    '6 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], l, k, n, m, C[l,k,n,m])
                assert abs(Voigt[i,j]-C[n,m,l,k]) < tol, \
                    '7 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i,j], n, m, l, k, C[n,m,l,k])

    return Voigt