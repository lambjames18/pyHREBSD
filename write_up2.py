import numpy as np
from collections import namedtuple
import struct
import os
# from tqdm.auto import tqdm
# import matplotlib.pyplot as plt



########## USER INPUTS ##########

### List of all the up2 filepaths that need to be processed
paths = ["E:/SiGe/a-C03-scan/ScanA_2048x2048.up2", "E:/SiGe/b-C04-scan/ScanB_2048x2048.up2"]

### List of the save paths for the processed up2 files
save_paths = ["E:/SiGe/a-C03-scan/ScanA_1024x1024.up2", "E:/SiGe/b-C04-scan/ScanB_1024x1024.up2"]

### Processing parameters
bin_pats = True        # Whether to bin the patterns
binning = 2            # Binning factor
binning_mode = "mean"  # Binning mode, "mean" or "sum"

flip_pats = True       # Whether to flip the patterns
flip = "lr"            # Flip mode, "lr", "ud", or "both"

######### END USER INPUTS #########





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
    # print("Header:", FirstEntryUpFile, sz1, sz2, bitsPerPixel)
    sizeBytes = os.path.getsize(up2) - 16
    sizeString = str(round(sizeBytes / 1e6, 1)) + " MB"
    bytesPerPixel = 2
    nPatternsRecorded = int((sizeBytes/bytesPerPixel) / (sz1 * sz2))
    out = namedtuple("up2_file", ["patshape", "filesize", "nPatterns", "datafile"])
    out = out((sz1, sz2), sizeString, nPatternsRecorded, upFile)
    return out


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
    elif isinstance(idx, np.ndarray):
        reshape = False
        if idx.ndim >= 2:
            reshape = True
            out_shape = idx.shape + pat_obj.patshape
            idx = idx.flatten()
    else:
        reshape = False

    # Read in the patterns
    start_byte = np.int64(16)
    pattern_bytes = np.int64(pat_obj.patshape[0] * pat_obj.patshape[1] * 2)
    pats = np.zeros((len(idx), *pat_obj.patshape), dtype=np.uint16)
    # for i in tqdm(range(len(idx)), desc="Reading patterns"):
    for i in range(len(idx)):
        pat = np.int64(idx[i])
        seek_pos = start_byte + pat * pattern_bytes
        pat_obj.datafile.seek(seek_pos)
        pats[i] = np.frombuffer(pat_obj.datafile.read(pat_obj.patshape[0] * pat_obj.patshape[1] * 2), dtype=np.uint16).reshape(pat_obj.patshape)

    # Reshape the patterns
    if reshape:
        pats = pats.reshape(out_shape)

    return pats


def write_up2(pats_array: np.ndarray, filename: str, bit_depth: int = 16):
    """Write a 3D array of patterns to a up2 file.

    Args:
        pats_array (np.ndarray): 3D array of patterns.
        filename (str): Path to save the up2 file.
        bit_depth (int): Bit depth of the patterns. Default is 16. can be 8 (up1) or 16 (up2)."""
    # Ensure pattern dimensions are correct
    if pats_array.ndim == 2:
        pats_array = pats_array[None, :, :]
    elif pats_array.ndim == 4:
        pats_array = pats_array.reshape(-1, pats_array.shape[-2], pats_array.shape[-1])
    elif pats_array.ndim != 3:
        raise ValueError("pats_array must be 2D (single pattern), 3D (an number of patterns), or 4D (a grid of patterns).")

    # Normalize the patterns and convert to uint8 or uint16
    mns = pats_array.min(axis=(-2, -1))[:, None, None]
    mxs = pats_array.max(axis=(-2, -1))[:, None, None]
    if bit_depth == 8:
        pats_array = np.around((pats_array - mns) / (mxs - mns) * 255).astype(np.uint8)
    elif bit_depth == 16:
        pats_array = np.around((pats_array - mns) / (mxs - mns) * 65535).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16.")

    # Check the file extension
    if ".up2" not in filename and ".up1" not in filename:
        filename += ".up2" if bit_depth == 16 else ".up1"
        print("Filename extension not recognized. Defaulting to", filename[-4:], "based on bit depth.")
    elif bit_depth == 16 and not filename.endswith(".up2"):
        filename = filename.replace(".up1", ".up2")
        print("Filename extension did not match bit depth. Changed to", filename[-4:])
    elif bit_depth == 8 and not filename.endswith(".up1"):
        filename = filename.replace(".up2", ".up1")
        print("Filename extension did not match bit depth. Changed to", filename[-4:])

    # Get the pattern dimensions
    sz1, sz2 = pats_array.shape[1:]

    # Open the file
    upFile = open(filename, "wb")

    # Write the header
    upFile.write(struct.pack('i', 1))
    upFile.write(struct.pack('i', sz1))
    upFile.write(struct.pack('i', sz2))
    upFile.write(struct.pack('i', bit_depth))

    # Write the patterns
    # for i in tqdm(range(pats_array.shape[0]), desc="Writing patterns"):
    for i in range(pats_array.shape[0]):
        upFile.write(pats_array[i].reshape(-1).tobytes())

    # Close the file
    upFile.close()


def bin_patterns(patterns: np.ndarray, binning: int, mode: str = "mean") -> np.ndarray:
    """Bin patterns by averaging or summing neighboring pixels.

    Args:
        patterns (np.ndarray): 3D array of patterns.
        binning (int): Factor to bin the patterns by.
        mode (str): Mode to bin the patterns. Options are "mean" or "sum". Default is "mean".

    Returns:
        np.ndarray: Binned patterns."""
    # Ensure the binning factor is valid
    if binning < 2 or not isinstance(binning, int) or binning % 2 != 0:
        raise ValueError("binning must be an integer multiple of 2.")

    # Get the pattern dimensions
    M, N = patterns.shape[-2:]

    # Get the binned dimensions
    MK = M // binning
    NL = N // binning

    # Reshape the patterns
    p = patterns[:, :MK*binning, :NL*binning].reshape(patterns.shape[0], MK, binning, NL, binning)

    # Bin the patterns
    if mode == "mean":
        pats_binned = p.mean(axis=(2, 4))
    elif mode == "sum":
        pats_binned = p.sum(axis=(2, 4))
    elif mode == "max":
        pats_binned = p.max(axis=(2, 4))
    elif mode == "min":
        pats_binned = p.min(axis=(2, 4))
    else:
        raise ValueError("mode must be 'mean', 'sum', 'max', or 'min'.")

    return pats_binned


def flip_patterns(patterns: np.ndarray, mode: str = None) -> np.ndarray:
    """Flip patterns along the horizontal or vertical axis.

    Args:
        patterns (np.ndarray): 3D array of patterns.
        mode (str): Mode to flip the patterns. Options are "lr", "ud", or "both". Default is None (no flipping).

    Returns:
        np.ndarray: Flipped patterns."""
    if mode is None:
        return patterns
    elif mode == "lr":
        return np.flip(patterns, axis=-1)
    elif mode == "ud":
        return np.flip(patterns, axis=-2)
    elif mode == "both":
        return np.flip(np.flip(patterns, axis=-1), axis=-2)
    else:
        raise ValueError("mode must be 'lr' or 'ud'.")


# Actually do the work
for i in range(len(paths)):
    print("Processing", paths[i])
    pat_obj = read_up2(paths[i])
    pats = get_patterns(pat_obj)

    if bin_pats:
        pats_out = bin_patterns(pats, binning, mode=binning_mode)
    if flip_pats:
        pats_out = flip_patterns(pats_out, mode=flip)
    write_up2(pats_out, save_paths[i], bit_depth=16)
    print("\tSaved to", save_paths[i])
