import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist

import pyHREBSD
import utilities

if __name__ == "__main__":
    t0 = time.time()

    ### Parameters ###
    # Names and paths
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

    # Set the homography center properly
    if h_center == "pattern":
        PC = ang_data.pc
    elif h_center == "image":
        PC = (pat_obj.patshape[1] / 2, pat_obj.patshape[0] / 2, ang_data.pc[2])

    if calc:
        # Get initial guesses
        R = pats[x0]
        T = pats[1:]
        tilt = 90 - sample_tilt + detector_tilt
        p0 = pyHREBSD.get_initial_guess(R, T, PC, tilt, initial_subset_size, guess_type)
        print(p0)

        # Get homographies
        subset_slice = (slice(int(PC[1] - subset_size[0] / 2), int(PC[1] + subset_size[0] / 2)),
                        slice(int(PC[0] - subset_size[1] / 2), int(PC[0] + subset_size[1] / 2)))
        p, i_count, residuals = pyHREBSD.get_homography(
            R,
            T,
            subset_slice=subset_slice,
            p0=p0,
            PC=PC,
            max_iter=max_iter,
            conv_tol=conv_tol,
            parallel=False,
        )
        print(p, i_count, residuals)
        np.save(f"{save_name}_p.npy", p)

    else:
        p = np.load(f"{save_name}_p.npy")

    # Get deformation gradients and strain
    Fe = pyHREBSD.homography_to_elastic_deformation(p, PC)
    e, w = pyHREBSD.deformation_to_stress_strain(Fe)

    # Print time
    t1 = time.time()
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f">Total time for the entire run: {hours:2.0f}:{minutes:2.0f}:{seconds:2.0f}")

    # Save the results
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(idx, e[:, 0, 0], label="e_xx")
    ax.plot(idx, e[:, 1, 1], label="e_yy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Strain")
    ax.legend()
    plt.tight_layout()
    fig1, ax1 = plt.subplots(2, 4, figsize=(16, 8))
    ax1[0, 0].plot(idx, p[:, 0])
    ax1[0, 1].plot(idx, p[:, 1])
    ax1[0, 2].plot(idx, p[:, 2])
    ax1[0, 3].plot(idx, p[:, 3])
    ax1[1, 0].plot(idx, p[:, 4])
    ax1[1, 1].plot(idx, p[:, 5])
    ax1[1, 2].plot(idx, p[:, 6])
    ax1[1, 3].plot(idx, p[:, 7])
    ax1[0, 0].set_title("p11")
    ax1[0, 1].set_title("p12")
    ax1[0, 2].set_title("p13")
    ax1[0, 3].set_title("p21")
    ax1[1, 0].set_title("p22")
    ax1[1, 1].set_title("p23")
    ax1[1, 2].set_title("p31")
    ax1[1, 3].set_title("p32")
    plt.tight_layout()
    plt.show()
