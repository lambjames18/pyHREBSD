import numpy as np
from scipy import signal, ndimage, interpolate


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
        # mask = (x >= self.xrange[0]) & (x <= self.xrange[1]) & (y >= self.yrange[0]) & (y <= self.yrange[1])
        # noise_range = np.percentile(out, (0.0, 1.0))
        # out[~mask] = np.random.uniform(noise_range[0], noise_range[1], np.sum(~mask))
        if normalize:
            return self.__normalize(out)
        return out

    def __normalize(self, a):
        mean = a.mean()
        return (a - mean) / np.sqrt(((a - mean)**2).sum())

    def warp(self, Wp):
        xi_3d = np.vstack((self.coords, np.ones(self.coords.shape[1])))
        xi_prime = Wp.dot(xi_3d)
        return self(xi_prime[0], xi_prime[1])

    def gradient(self):
        dx = self(self.coords[0], self.coords[1], dx=1, dy=0, grid=False, normalize=False)
        dy = self(self.coords[0], self.coords[1], dx=0, dy=1, grid=False, normalize=False)
        dxy = np.vstack((dx, dy)).reshape(2, 1, -1).transpose(1, 0, 2)  # 2x1xN
        return dxy


def get_GR(R, subset_slice, PC):
    # Get coordinates
    x = np.arange(R.shape[1]) - PC[0]
    y = np.arange(R.shape[0]) - PC[1]
    X, Y = np.meshgrid(x, y)
    xi = np.array([Y[subset_slice].flatten(), X[subset_slice].flatten()])

    # Compute the intensity gradients of the subset
    spline = interpolate.RectBivariateSpline(x, y, R, kx=5, ky=5)
    GRx = spline(xi[0], xi[1], dx=1, dy=0, grid=False)
    GRy = spline(xi[0], xi[1], dx=0, dy=1, grid=False)
    GR = np.vstack((GRx, GRy)).reshape(2, 1, -1).transpose(1, 0, 2)
    return GR


if __name__ == "__main__":
    import utilities
    import matplotlib.pyplot as plt
    import pyHREBSD
    import timeit

    save_name = f"SiGeScanA"
    up2 = "E:/SiGe/ScanA.up2"
    ang = "E:/SiGe/ScanA.ang"

    # Geometry
    pixel_size = 13.0  # The pixel size in um
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 10.1  # The detector tilt in degrees

    # Pattern processing
    truncate = True
    equalize = False
    DoG_sigmas = (1.0, 30.0)  # The sigmas for the difference of Gaussians filter

    # Initial guess
    initial_subset_size = 2048  # The size of the subset, must be a power of 2
    guess_type = "partial"  # The type of initial guess to use, "full", "partial", or "none"

    # Subpixel registration
    h_center = "image"  # The homography center for deformation, "pattern" or "image"
    max_iter = 50  # The maximum number of iterations for the subpixel registration
    conv_tol = 1e-4  # The convergence tolerance for the subpixel registration
    subset_shape = "rectangle"  # The shape of the subset for the subpixel registration, "rectangle", "ellipse", or "donut"
    subset_size = (1638, 1638) # The size of the subset for the subpixel registration, (H, W) for "rectangle", (a, b) for "ellipse", or (r_in, r_out) for "donut"

    # Reference index
    x0 = 0  # The index of the reference pattern

    # Run the calc or load the results
    calc = True
    ### Parameters ###


    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang)

    idx = np.arange(75, 95, 5)
    idx = np.array([0, 1])
    pats = utilities.get_patterns(pat_obj, idx=idx).astype(float)
    pats = utilities.process_patterns(pats, equalize=equalize, truncate=truncate)
    PC = (1024, 1024)

    subset_slice = (slice(int(PC[1] - subset_size[0] / 2), int(PC[1] + subset_size[0] / 2)),
                    slice(int(PC[0] - subset_size[1] / 2), int(PC[0] + subset_size[1] / 2)))

    spline = Spline(pats[0], 5, 5, ang_data.pc, subset_slice)
    t0 = timeit.timeit(lambda: spline.gradient(), number=10) / 10
    t1 = timeit.timeit(lambda: get_GR(pats[0], subset_slice, ang_data.pc), number=10) / 10
    print(f"Spline: {t0:.2e} s")
    print(f"GR: {t1:.2e} s")


