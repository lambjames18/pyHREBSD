import numpy as np
from collections import namedtuple
from scipy import ndimage
from skimage import exposure
import re
from tqdm.auto import tqdm

import ebsd_pattern as ep
import rotations

NUMERIC = r"[-+]?\d*\.\d+|\d+"


def read_up2(up2: str) -> np.ndarray:
    """Read in patterns and a pattern center from an ang file and a pattern file.
    Only supports EDAX/TSL.

    Args:
        up2 (str): Path to the pattern file."""
    # Get patterns
    obj = ep.get_pattern_file_obj(up2)
    obj.read_header()
    pats = obj.read_data(returnArrayOnly=True)
    return pats


def read_ang(path: str) -> namedtuple:
    """Reads in the pattern center from an ang file.
    Only supports EDAX/TSL.

    Args:
        ang (str): Path to the ang file."""
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
    PC = (xstar, ystar, zstar)
    shape = (rows, cols)
    names.extend(["eulers", "quats", "shape", "pc"])
    names = [name.replace(" ", "_").lower() for name in names if name.lower() not in ["phi1", "phi", "phi2"]]

    # Read in the data
    ang_data = np.genfromtxt(path, skip_header=header_lines)
    ang_data = ang_data.reshape(shape + (ang_data.shape[1],))
    euler = ang_data[..., 0:3]
    ang_data = ang_data[..., 3:]
    qu = rotations.eu2qu(euler)
    ang_data = np.moveaxis(ang_data, 2, 0)

    # Package everything into a namedtuple
    ang_data = namedtuple("ang_file", names)(*ang_data, eulers=euler, quats=qu, shape=shape, pc=PC)
    return ang_data


def get_scan_data(up2: str, ang: str, Nxy: tuple, pixel_size: float = 10.0, calculate_sharpness: bool = False) -> tuple:
    """Reads in patterns and orientations from an ang file and a pattern file.
    Only supports EDAX/TSL.
    
    Args:
        up2 (str): Path to the pattern file.
        ang (str): Path to the ang file.
        Nxy (tuple): The detector dimensions before binning. Used for converting the pattern center.
        pixel_size (float): The detector pixel size. Used for converting the pattern center.
        calculate_sharpness (bool): Whether to calculate the sharpness of the patterns.

    Returns:
        np.ndarray: The patterns.
        namedtuple: The orientations. namedtuple with fields corresponding to the columns in the ang file + eulers, quats, shape, pc.
                    Sharpness replaces image quality (iq) if calculate_sharpness is True."""
    # Get the patterns
    ### TODO: Add lazy loading for large files
    pats = read_up2(up2)

    # Get the ang data
    ang_data = read_ang(ang)

    # Reshape patterns and scale them to (0, 1) range
    pats = pats.reshape(ang_data.shape + pats.shape[1:]).astype(np.float32)
    mins = pats.min(axis=(2, 3))[:, :, None, None]
    maxs = pats.max(axis=(2, 3))[:, :, None, None]
    pats = (pats - mins) / (maxs - mins)

    # Convert the pattern center
    PC = convert_pc(ang_data.pc, Nxy, pixel_size)
    ang_data = ang_data._replace(pc=PC)

    # Calculate sharpness
    if calculate_sharpness:
        shp = get_sharpness(pats)
        ang_data = ang_data._replace(iq=shp)

    return pats, ang_data


def convert_pc(PC: tuple, N: tuple, delta: float, b: float = 1.0) -> tuple:
    """
    Converts the pattern center from EDAX/TSL standard to the EMsoft standard
    Input:
        PC: tuple of floats (xstar, ystar, zstar)
        Nsy: tuple, detector dimensions before binning
        delta: float, the detector pixel size
        b: 
    Output:
        PC: tuple of floats (xpc, ypc, L)"""
    xpc = np.around(N[0] * (PC[0] - 0.5), 4)
    ypc = np.around(N[0] * PC[1] - b * N[1] * 0.5, 4)
    L = np.around(N[0] * delta * PC[2], 4)
    return (xpc, ypc, L)


def get_sharpness(imgs: np.ndarray) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.
    
    Args:
        imgs (np.ndarray): The images to calculate the sharpness of. (H, W) or (N, H, W) or (N0, N1, H, W)

    Returns:
        np.ndarray: The sharpness of the images. float, (N,) or (N0, N1) for example."""
    if imgs.ndim == 3:
        imgs = imgs[:, None, ...]
    elif imgs.ndim == 2:
        imgs = imgs[None, None, ...]
    shp = np.zeros((imgs.shape[0], imgs.shape[1]))
    for i in tqdm(range(imgs.shape[0]), desc='Calculating sharpness', leave=False, position=0, unit='imgs'):
        f = np.fft.fft2(imgs[i], axes=(1, 2))
        f = np.real(f)
        f = np.fft.fftshift(f, axes=(1, 2))
        AF = abs(f)
        M = AF.max(axis=(1, 2))
        thresh = M / 2500
        th = (f > thresh[:, None, None]).sum(axis=(1, 2))
        shp[i] = th / (imgs.shape[2] * imgs.shape[3])
    shp = np.squeeze(shp)
    if shp.shape == ():
        shp = shp.item()
    return shp


def clean_patterns(pats: np.ndarray, background: bool = True, equalize: bool = True, gauss_sigma: int = 1) -> np.ndarray:
    """Cleans patterns by subtracting the background, normalizing the intensity.
    Optionally, applies a Gaussian filter and/or applies adaptive histogram equalization.

    Args:
        pats (np.ndarray): The patterns to clean. (N, H, W)
        background (bool): Whether to subtract the background.
        equalize (bool): Whether to apply adaptive histogram equalization.
        gauss_sigma (float): The standard deviation of the Gaussian filter. If 0, no filter is applied.

    Returns:
        np.ndarray: The cleaned patterns. (N, H, W)"""
    background = ndimage.gaussian_filter(pats.mean(axis=(0,1)), 5)
    pats = pats - background[None, None]
    pats = (pats - pats.min(axis=(2,3))[:, :, None, None]) / (pats.max(axis=(2, 3)) - pats.min(axis=(2, 3)))[:, :, None, None]
    if gauss_sigma > 0:
        func_temp = lambda x: ndimage.gaussian_filter(x, gauss_sigma)
    else:
        func_temp = lambda x: x
    if equalize:
        func = lambda x: exposure.equalize_adapthist(func_temp(x), clip_limit=0.03)
    else:
        func = func_temp
    for i in tqdm(range(pats.shape[0]), desc='Cleaning patterns', leave=False, position=0, unit='patterns'):
        for j in range(pats.shape[1]):
            pats[i, j] = func(pats[i, j])
    return pats
