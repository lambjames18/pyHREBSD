import struct
import os
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp

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


def get_sharpness(ebsd_data: UP2,
                  batch_size: int = 8,
                  n_cpus: int = 1,
                  lazy = True) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.

    Args:
        ebsd_data (UP2 object): The data to calculate the sharpness of.
        batch_size (int): The number of images to process at once.
        n_cpus (int): The number of cpus to use.
        lazy (bool): Whether to use lazy loading or not. Usefull for large datasets.

    Returns:
        np.ndarray: The sharpness of the patterns. float, (N,)"""
    if not lazy:
        imgs = ebsd_data.read_patterns()
        # Put into batches
        imgs_split = np.array_split(imgs, imgs.shape[0] // batch_size)  # (M, batch_size, H, W) where M is the number of batches and batch_size is not a constant

        if n_cpus > 1:
            with mp.Pool(processes=n_cpus) as pool:
                shp = tqdm(pool.imap(_calc_sharpness_numpy, imgs_split), total=len(imgs_split))
                shp = np.concatenate(list(shp))
        else:
            shp = np.array([])
            for i in tqdm(range(len(imgs_split)), desc='Calculating sharpness', unit='batches'):
                shp = np.concatenate((shp, _calc_sharpness_numpy(imgs_split[i])))
    else:
        # Put into batches
        idx = np.arange(ebsd_data.nPatterns)
        idx_split = np.array_split(idx, ebsd_data.nPatterns // batch_size)  # (M, batch_size) where M is the number of batches and batch_size is not a constant
        args = [(ebsd_data, idx) for idx in idx_split]

        if n_cpus > 1:
            with mp.Pool(processes=n_cpus) as pool:
                shp = tqdm(pool.imap(_calc_sharpness_numpy_lazy, args), total=len(args))
                shp = np.concatenate(list(shp))
        else:
            shp = np.array([])
            for i in tqdm(range(len(idx_split)), desc='Calculating sharpness', unit='batches'):
                shp = np.concatenate((shp, _calc_sharpness_numpy_lazy(*args[i])))

    # Reshape if necessary
    shp = np.squeeze(shp)
    return shp


def _calc_sharpness_numpy_lazy(args) -> np.ndarray:
    """Calculates the sharpness of an image/stack of images.
    Arguments are passed as a tuple to allow for multiprocessing.

    Args:
        args[0]: ebsd_data (UP2 object): The data to calculate the sharpness of.
        args[1]: idx (np.ndarray): The indices of the images to calculate the sharp

    Returns:
        np.ndarray: The sharpness of the images. float, (batch_size,)"""
    ebsd_data, idx = args
    pats = ebsd_data.read_patterns(idx)
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


if __name__ == "__main__":
    up2_path = ...
    batch_size = 8
    lazy = True
    n_cpus = 10
    output_shape = (..., ...)
    
    out_path = os.path.join(
        os.path.dirname(up2_path),
        os.path.basename(up2_path).split(".")[-2] + "_sharpness.npy"
    )

    ebsd_data = UP2(up2_path)
    sharpness = get_sharpness(ebsd_data, batch_size=8, lazy=True, n_cpus=10).reshape(output_shape)
    print("Saving sharpness map to", out_path)
    np.save(out_path, sharpness)
