import numpy as np
import re

NUMERIC = r"[-+]?\d*\.\d+|\d+"

def convert_pc(PC: tuple, N: tuple, delta: float, b: float=1.0) -> tuple:
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

def fft_transform(array):
    imgarray = np.asarray(array, dtype='float')
    f = np.fft.fft2(imgarray)
    f = np.real(f)
    fshift = np.fft.fftshift(f)
    return fshift


def pattern_sharpness(pattern: np.ndarray, return_spectrum: bool=False) -> float | tuple[float, np.ndarray]:
    """Calculates the pattern sharpness via an FFT transform
    Input:
        pattern: 2D array of the pattern
        return_spectrum: bool, if True, returns the power spectrum
    Output:
        shp: float, the sharpness value
        log_pow_spect: (optional) 2D array, the log of the power spectrum"""
    y = fft_transform(pattern)
    AF = abs(y)
    M = AF.max()
    thresh = M/2500
    th = (y>thresh).sum()
    shp = th/(pattern.shape[0]*pattern.shape[1])
    log_pow_spect = np.log(AF)
    fshiftdisp = (y-y.min())/(y.max()-y.min())*255
    if return_spectrum:
        return (shp,log_pow_spect.astype('uint16'))
    else:
        return shp

def get_PC_from_ang(path):
    with open(path, "r") as ang:
        for line in ang:
            if "x-star" in line:
                xstar = float(re.findall(NUMERIC, line)[0])
            elif "y-star" in line:
                ystar = float(re.findall(NUMERIC, line)[0])
            elif "z-star" in line:
                zstar = float(re.findall(NUMERIC, line)[0])
            elif "HEADER: End" in line:
                break
    return (xstar, ystar, zstar)