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


class UP2:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.i = 0
        self.start_byte = np.int64(16)
        self.header()

    def header(self):
        chunk_size = 4
        tmp = self.read(chunk_size)
        self.FirstEntryUpFile = struct.unpack("i", tmp)[0]
        tmp = self.read(chunk_size)
        sz1 = struct.unpack("i", tmp)[0]
        tmp = self.read(chunk_size)
        sz2 = struct.unpack("i", tmp)[0]
        tmp = self.read(chunk_size)
        self.bitsPerPixel = struct.unpack("i", tmp)[0]
        sizeBytes = os.path.getsize(self.path) - 16
        self.filesize = str(round(sizeBytes / 1e6, 1)) + " MB"
        bytesPerPixel = 2
        self.nPatterns = int((sizeBytes / bytesPerPixel) / (sz1 * sz2))
        self.patshape = (sz1, sz2)
        self.pattern_bytes = np.int64(self.patshape[0] * self.patshape[1] * 2)

    def read(self, chunks, i=None):
        """Read the next `chunks` bytes from the file. If `i` is not None, read from the current position."""
        if i is None:
            i = self.i
        with open(self.path, "rb") as upFile:
            upFile.seek(i)
            data = upFile.read(chunks)
        self.i += chunks
        return data

    def read_pattern(self, i, process=False, p_kwargs={}):
        # Read in the patterns
        seek_pos = np.int64(self.start_byte + np.int64(i) * self.pattern_bytes)
        buffer = self.read(chunks=self.pattern_bytes, i=seek_pos)
        pat = np.frombuffer(buffer, dtype=np.uint16).reshape(self.patshape)
        if process:
            pat = self.process_pattern(pat, **p_kwargs)
        return pat

    def read_patterns(self, idx=-1, process=False, p_kwargs={}):
        if type(idx) == int:
            if idx != -1:
                return self.read_pattern(idx)
            else:
                idx = range(self.nPatterns)
        else:
            idx = np.asarray(idx)

        # Read in the patterns
        in_shape = idx.shape + self.patshape
        idx = idx.flatten()
        if process:
            pats = np.zeros(idx.shape + self.patshape, dtype=np.float32)
        else:
            pats = np.zeros(idx.shape + self.patshape, dtype=np.uint16)
        for i in range(idx.shape[0]):
            pats[i] = self.read_pattern(idx[i], process, p_kwargs)
        return pats.reshape(in_shape)

    def process_pattern(
            self,
            img: np.ndarray,
            low_pass_sigma: float = 2.5,
            high_pass_sigma: float = 101,
            truncate_std_scale: float = 3.0
        ) -> np.ndarray:
        """Cleans patterns by equalizing the histogram and normalizing.
        Args:
            img (np.ndarray): The patterns to clean. (H, W)
            low_pass_sigma (float): The sigma for the low pass filter.
            high_pass_sigma (float): The sigma for the high pass filter.
            truncate_std_scale (float): The number of standard deviations to truncate.
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


def process_pattern(
        img: np.ndarray,
        low_pass_sigma: float = 2.5,
        high_pass_sigma: float = 101,
        truncate_std_scale: float = 3.0
    ) -> np.ndarray:
    """Cleans patterns by equalizing the histogram and normalizing.
    Args:
        img (np.ndarray): The patterns to clean. (H, W)
        low_pass_sigma (float): The sigma for the low pass filter.
        high_pass_sigma (float): The sigma for the high pass filter.
        truncate_std_scale (float): The number of standard deviations to truncate.
    Returns:
        np.ndarray: The cleaned patterns. (N, H, W)"""
    
    # Process inputs
    img = img.astype(np.float32)
    # median = np.percentile(img, 50)
    # low_percentile = np.percentile(img, 0.1)
    # high_percentile = np.percentile(img, 99.9)
    # img[img < low_percentile] = median
    # img[img > high_percentile] = median

    # Low pass filter
    img = (img - img.min()) / (img.max() - img.min())
    img = ndimage.gaussian_filter(img, low_pass_sigma)

    # High pass filter
    background = ndimage.gaussian_filter(img, high_pass_sigma)
    img = img - background
    img = (img - img.min()) / (img.max() - img.min())

    # Truncate step
    if truncate_std_scale > 0:
        mean, std = img.mean(), img.std()
        img = np.clip(img, mean - truncate_std_scale * std, mean + truncate_std_scale * std)

    return img


if __name__ == "__main__":
    up2_path = "E:/SiGe/a-C03-scan/ScanA_1024x1024.up2"
    up2 = UP2(up2_path)
    pat = up2.read_pattern(0, process=True)
    pat = np.around(255 * (pat - pat.min()) / (pat.max() - pat.min())).astype(np.uint8)
    io.imsave("pattern.png", pat)
