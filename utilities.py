import numpy as np

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
