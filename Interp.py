import numpy as np
from scipy import signal, ndimage, interpolate
from skimage import exposure

import warp


class Spline:
    def __init__(self, image, kx, ky, PC=None, subset_slice=None):
        # Inputs
        self.image = image
        self.kx = kx
        self.ky = ky
        if PC is None:
            self.PC = np.array(image.shape)[::-1] / 2
        else:
            self.PC = PC
        if subset_slice is None:
            self.subset_slice = (slice(None), slice(None))
        else:
            self.subset_slice = subset_slice

        # Create the spline
        x = np.arange(image.shape[1]) - PC[0]
        y = np.arange(image.shape[0]) - PC[1]
        self.xrange = (x[0], x[-1])
        self.yrange = (y[0], y[-1])
        X, Y = np.meshgrid(x, y)
        self.coords = np.array([Y[subset_slice].flatten(), X[subset_slice].flatten()])
        self.S = interpolate.RectBivariateSpline(x, y, image, kx=kx, ky=ky)

    def __call__(self, x, y, dx=0, dy=0, grid=False, normalize=True):
        out = self.S(x, y, dx=dx, dy=dy, grid=grid)
        mask = (x >= self.xrange[0]) & (x <= self.xrange[1]) & (y >= self.yrange[0]) & (y <= self.yrange[1])
        noise_range = np.percentile(out, (0.0, 1.0))
        out[~mask] = np.random.uniform(noise_range[0], noise_range[1], np.sum(~mask))
        if normalize:
            return self.__normalize(out)
        return out

    def __normalize(self, a):
        mean = a.mean()
        return (a - mean) / np.sqrt(((a - mean)**2).sum())

    def warp(self, h, normalize=True):
        Wp = warp.W(h)
        xi_3d = np.vstack((self.coords, np.ones(self.coords.shape[1])))
        xi_prime = Wp.dot(xi_3d)
        return self(xi_prime[0], xi_prime[1], normalize=normalize)

    def gradient(self):
        dx = self(self.coords[0], self.coords[1], dx=1, dy=0, grid=False, normalize=False)
        dy = self(self.coords[0], self.coords[1], dx=0, dy=1, grid=False, normalize=False)
        dxy = np.vstack((dx, dy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 1x2xN
        return dxy
