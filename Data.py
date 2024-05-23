import os
import struct
from collections import namedtuple
import re

from skimage import io, exposure
from scipy import ndimage
from tqdm import tqdm
import numpy as np

import rotations


NUMERIC = r"[-+]?\d*\.\d+|\d+"


class EBSDData:
    """Class to hold EBSD data and provide methods to process it.
    It is assumed that the data is in the EDAX/TSL format.
    This holds both the scan information (ang file) and the patterns (up2 file).
    The scan data is read in during initialization, and the patterns are read in when needed.
    Upon reading in a pattern, the pattern is processed and stored in memory.
    To set, the set_processing_parameters method should be used.
    To get a single pattern, the get_pattern method should be used.
    To get multiple patterns, the get_patterns method should be used.
    """

    def __init__(self, up2: str, ang: str):
        """Initializes the EBSDData object.

        Args:
            up2 (str): Path to the pattern file.
            ang (str): Path to the ang file."""
        self.pat_obj, self.ang_data = get_scan_data(up2, ang)
        for attr in self.ang_data._fields:
            setattr(self, attr, getattr(self.ang_data, attr))
        for attr in self.pat_obj._fields:
            setattr(self, attr, getattr(self.pat_obj, attr))
        self.sigma = 0.0
        self.equalize = False
        self.truncate = False

    def set_processing_parameters(self, sigma: float = 0.0, equalize: bool = True, truncate: bool = True):
        """Sets the processing parameters for the patterns.

        Args:
            sigma (float): The sigma for the Gaussian filter.
            equalize (bool): Whether to equalize the histogram.
            truncate (bool): Whether to truncate the patterns."""
        self.sigma = sigma
        self.equalize = equalize
        self.truncate = truncate

    def get_patterns(self, idx: int | np.ndarray | tuple | list) -> np.ndarray:
        """Gets a single pattern.

        Args:
            idx (int | np.ndarray | tuple | list): The index of the pattern to get.

        Returns:
            np.ndarray: The pattern."""
        if isinstance(idx, (int, np.integer)):
            idx = np.array([idx])
        else:
            idx = np.asarray(idx)
        patterns = get_patterns(self.pat_obj, idx)
        patterns = process_patterns(patterns, self.sigma, self.equalize, self.truncate)
        return patterns


def process_patterns(img: np.ndarray, sigma: float = 0.0, equalize: bool = True, truncate: bool = True) -> np.ndarray:
    """Cleans patterns by equalizing the histogram and normalizing.

    Args:
        img (np.ndarray): The patterns to clean. (H, W)
        equalize (bool): Whether to equalize the histogram.
        high_pass (bool): Whether to apply a high-pass filter.
        truncate (bool): Whether to truncate the patterns.

    Returns:
        np.ndarray: The cleaned patterns. (N, H, W)"""
    # Process inputs
    if img.ndim == 2:
        img = img.reshape(1, img.shape[0], img.shape[1])
    img = img.astype(np.float32)
    median = np.percentile(img, 50)
    low_percentile = np.percentile(img, 1)
    high_percentile = np.percentile(img, 99)

    # Process the patterns
    for i in range(img.shape[0]):
        if truncate:
            img[i][img < low_percentile] = median
            img[i][img > high_percentile] = median
        img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        background = ndimage.gaussian_filter(img[i].mean(axis=0), img[i].shape[-1] / 10)
        img[i] = img[i] - background
        img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        if equalize:
            img[i] = exposure.equalize_adapthist(img)
            img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        if sigma > 0.0:
            img[i] = ndimage.gaussian_filter(img, sigma)
            img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())

    return img

def convert_pc(PC: tuple | list | np.ndarray, patshape: tuple | list | np.ndarray) -> tuple:
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


def read_ang(path: str, patshape: tuple | list | np.ndarray) -> namedtuple:
    """Reads in the pattern center from an ang file.
    Only supports EDAX/TSL.

    To print the data columns in the ang file, use the following:
    >>> ang_data = read_ang("path/to/ang/file.ang")
    >>> print(ang_data._fields)

    Args:
        ang (str): Path to the ang file.
        patshape (tuple): The shape of the patterns.

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
    PC = convert_pc((xstar, ystar, zstar), patshape)
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
    out = namedtuple("ang_file", names)(*ang_data, eulers=euler, quats=qu, shape=shape, pc=PC, pidx=pidx)
    return out


def get_scan_data(up2: str, ang: str) -> tuple:
    """Reads in patterns and orientations from an ang file and a pattern file.
    Only supports EDAX/TSL.
    
    Args:
        up2 (str): Path to the pattern file.
        ang (str): Path to the ang file.

    Returns:
        np.ndarray: The patterns.
        namedtuple: The orientations. namedtuple with fields corresponding to the columns in the ang file + eulers, quats, shape, pc."""
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
    for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
        pat = np.int64(idx[i])
        seek_pos = start_byte + pat * pattern_bytes
        pat_obj.datafile.seek(seek_pos)
        pats[i] = np.frombuffer(pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2), dtype=np.uint16).reshape(pat_obj.patshape)

    # Reshape the patterns
    if reshape:
        pats = pats.reshape(out_shape)

    return pats


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
        print(" -- get_index warning: the region of interest is too large in the x-direction given the center point.")
    if y1 - y0 < size:
        print(" -- get_index warning: the region of interest is too large in the y-direction given the center point.")
    idx = ang_data.pidx[x0:x1, y0:y1]
    return idx
