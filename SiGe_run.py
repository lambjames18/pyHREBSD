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
    save_name = f"SiGeScanB"
    up2 = "E:/SiGe/ScanB.up2"
    ang = "E:/SiGe/ScanB.ang"

    # Geometry
    pixel_size = 13.0  # The pixel size in um
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 10.1  # The detector tilt in degrees

    # Pattern processing
    truncate = True
    sigma = 20
    equalize = True

    # Initial guess
    initial_subset_size = 2048  # The size of the subset, must be a power of 2
    guess_type = "partial"  # The type of initial guess to use, "full", "partial", or "none"

    # Subpixel registration
    h_center = "image"  # The homography center for deformation, "pattern" or "image"
    max_iter = 50  # The maximum number of iterations for the subpixel registration
    conv_tol = 1e-3  # The convergence tolerance for the subpixel registration
    subset_shape = "rectangle"  # The shape of the subset for the subpixel registration, "rectangle", "ellipse", or "donut"
    subset_size = (2000, 2000) # The size of the subset for the subpixel registration, (H, W) for "rectangle", (a, b) for "ellipse", or (r_in, r_out) for "donut"

    # Reference index
    x0 = 0  # The index of the reference pattern

    # Run the calc or load the results
    calc = True
    ### Parameters ###


    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang)

    idx = np.arange(0, pat_obj.nPatterns)

    # Set the homography center properly
    if h_center == "pattern":
        PC = ang_data.pc
    elif h_center == "image":
        PC = (pat_obj.patshape[1] / 2, pat_obj.patshape[0] / 2, ang_data.pc[2])

    if calc:
        # Get patterns
        pats = utilities.get_patterns(pat_obj, idx=idx).astype(float)
        pats = utilities.process_patterns(pats, sigma=sigma, equalize=equalize,truncate=truncate)

        # Get initial guesses
        R = pats[x0]
        T = pats
        tilt = 90 - sample_tilt + detector_tilt
        p0 = pyHREBSD.get_initial_guess(R, T, PC, tilt, initial_subset_size, guess_type)

        # Get homographies
        subset_slice = (slice(int(PC[1] - subset_size[0] / 2), int(PC[1] + subset_size[0] / 2)),
                        slice(int(PC[0] - subset_size[1] / 2), int(PC[0] + subset_size[1] / 2)))
        print("Getting homographies...")
        p, i_count, residuals = pyHREBSD.get_homography(
            R,
            T,
            subset_slice=subset_slice,
            p0=p0,
            PC=PC,
            max_iter=max_iter,
            conv_tol=conv_tol,
            parallel=True,
        )
        np.save(f"{save_name}_p.npy", p)
        np.save(f"{save_name}_i_count.npy", i_count)
        np.save(f"{save_name}_residuals.npy", residuals)

    else:
        p = np.load(f"{save_name}_p.npy")
        i_count = np.load(f"{save_name}_i_count.npy")
        residuals = np.load(f"{save_name}_residuals.npy")

    # Get deformation gradients and strain
    # PC_mod = (ang_data.pc[0] - 1024, ang_data.pc[1] - 1024, PC[2])
    PC_mod = ((ang_data.pc[0] - 1024) * pixel_size, (ang_data.pc[1] - 1024) * pixel_size, PC[2] * pixel_size)
    Fe = pyHREBSD.homography_to_elastic_deformation(p, PC_mod)
    # C = utilities.get_stiffness_tensor(165.6, 63.9, 79.5, structure="cubic")
    # e, w, s = pyHREBSD.deformation_to_stress_strain(Fe, C, small_strain=False)
    e, w, s = pyHREBSD.deformation_to_stress_strain(Fe, small_strain=False)

    # Print time
    t1 = time.time()
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f">Total time for the entire run: {hours:2.0f}:{minutes:2.0f}:{seconds:2.0f}")

    # Save the results
    fig, ax = plt.subplots(3, 3, figsize=(15, 14))
    ax[0, 0].scatter(idx, e[..., 0, 0], marker="s", label=r"$\epsilon_{11}$")
    ax[0, 1].scatter(idx, e[..., 0, 1], marker="s", label=r"$\epsilon_{12}$")
    ax[0, 2].scatter(idx, e[..., 0, 2], marker="s", label=r"$\epsilon_{13}$")
    ax[1, 1].scatter(idx, e[..., 1, 1], marker="s", label=r"$\epsilon_{22}$")
    ax[1, 2].scatter(idx, e[..., 1, 2], marker="s", label=r"$\epsilon_{23}$")
    ax[2, 2].scatter(idx, e[..., 2, 2], marker="s", label=r"$\epsilon_{33}$")

    for a in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2], ax[2, 2]]:
        a.set_ylim(-0.01, 0.01)
        # a.set_ylim(-0.0016, 0.0016)
        utilities.standardize_axis(a)
        utilities.make_legend(a)

    ax[1, 0].scatter(idx, i_count, marker="s", label="Iterations")
    ax[2, 0].scatter(idx, residuals, marker="s", label="Residuals")
    # ax[1, 0].axis("off")
    # ax[2, 0].axis("off")
    ax[2, 1].axis("off")
    for a in [ax[1, 0], ax[2, 0]]:
        utilities.standardize_axis(a)
        utilities.make_legend(a)

    plt.tight_layout()
    plt.savefig(f"{save_name}_strain_results.png", dpi=300)