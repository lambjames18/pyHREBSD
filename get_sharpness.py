import os
import numpy as np
import matplotlib.pyplot as plt
import ebsd_pattern
from tqdm.auto import tqdm
from skimage import io
import mpire
import h5py


def fft_transform(img):
    imgarray = np.asarray(img, dtype='float')
    f = np.fft.fft2(imgarray)
    f = np.real(f)
    fshift = np.fft.fftshift(f)
    return fshift

def Sharpness(img, return_pwr_spct=False):
    y = fft_transform(img)
    AF = abs(y)
    M = AF.max()
    thresh = M / 2500
    th = (y > thresh).sum()
    shp = th / (img.shape[0] * img.shape[1])
    if return_pwr_spct:
        log_pow_spect = np.log(AF)
        log_pow_spect = (log_pow_spect - log_pow_spect.min()) / (log_pow_spect.max() - log_pow_spect.min()) * 65535
        fshiftdisp = (y - y.min()) / (y.max() - y.min()) * 255
        return (shp,log_pow_spect.astype('uint16'))
    else:
        return shp

def fft_transform_multi(imgs):
    imgarray = np.asarray(imgs, dtype='float')
    f = np.fft.fft2(imgarray, axes=(1,2))
    f = np.real(f)
    fshift = np.fft.fftshift(f, axes=(1,2))
    return fshift

def Sharpness_multi(imgs):
    ys = fft_transform_multi(imgs)
    AF = abs(ys)
    M = AF.max(axis=(1,2))
    thresh = M / 2500
    th = (ys > thresh[:,None,None]).sum(axis=(1,2))
    shp = th / (imgs.shape[1] * imgs.shape[2])
    return shp

def main(indices, pobj):
    shp = []
    for index in indices:
        p = np.squeeze(pobj.pat_reader(index, 1))
        shp.append(Sharpness(p))
    return shp

def main_multi(pats):
    shp = Sharpness_multi(pats)
    return shp

def read_H5(path):
    h5 = h5py.File(path, 'r')
    pats = h5["EMData/DefectEBSD/EBSDPatterns"][:]
    h5.close()
    shape = pats.shape[:2]
    pats = pats.reshape(-1, *pats.shape[2:])
    return pats, shape

def read_UP2(path, shape):
    pobj = ebsd_pattern.get_pattern_file_obj(path)
    pobj.read_header()
    pats = np.squeeze(pobj.pat_reader(0, pobj.nPatterns))
    return pats, shape


if __name__ == "__main__":

    # file = "E:/James/20230906_24015_10kV_3200pA_1200x1200-1pointrepeat_120HFW_100nmstepsize_10point5cameratilt_11WD_movie_256x256.up2"
    # file = "Y:/Archive/2023/James/James_90degSCANSTRATEGY/20230906_24015_10kV_3200pA_1200x1200-1pointrepeat_120HFW_100nmstepsize_10point5cameratilt_11WD_movie_1024x1024.up2"
    # file = "E:/Evan/Print14_block-LC-11/Area 2/map20231107135457056.up2"
    # pats, shape = read_UP2(file, (1501, 1001))
    file = "F:/MDG_SimulatedDislocationDataset/EBSDoutfullRAM.h5"
    dirname, filename = os.path.split(file)

    print(f"Reading in {filename}")
    pats, shape = read_H5(file)
    print(f"Finished reading in {filename}, shape: {shape}, number of patterns: {pats.shape}")

    batches = pats.shape[0] // 150
    print(f"Calculating sharpness in {batches} batches")
    pats_split = np.array_split(pats, batches)
    
    with mpire.WorkerPool(n_jobs=24) as pool:
        sharpness = pool.map(main_multi, pats_split, progress_bar=True)

    sharpness = sharpness.reshape(shape)
    print(f"Saving sharpness to {dirname}")
    io.imsave(os.path.join(dirname, filename.split(".")[0] + "_raw.tiff"), sharpness)
    sharpness = np.around((sharpness - sharpness.min()) / (sharpness.max() - sharpness.min()) * 65535).astype('uint16')
    io.imsave(os.path.join(dirname, filename.split(".")[0] + "_uint16.tiff"), sharpness)
    print("Done")
