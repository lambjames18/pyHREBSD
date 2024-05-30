import time
import numpy as np
import matplotlib.pyplot as plt

import rotations
import utilities
import get_homography
import conversions

if __name__ == "__main__":
    ############################
    # Load the pattern object
    sample = "A"  # The sample to analyze, "A" or "B"
    name = dict(A="SiGeScanA_test", B="SiGeScanB_test")[sample]
    up2 = dict(A="E:/SiGe/ScanA.up2", B="E:/SiGe/ScanB.up2")[sample]
    ang = dict(A="E:/SiGe/ScanA.ang", B="E:/SiGe/ScanB.ang")[sample]
    # Set the geometry parameters
    pixel_size = 13.0  # The pixel size in um, taking binning into account (so 4xpixel_size for 4x4 binning)
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 10.1  # The detector tilt in degrees
    step_size = 0.450  # The step size in um
    subset_size = 1600
    initial_guess_subset_size = 1024
    # Set the roi parameters
    start = (0, 0)  # The pixel location to start the ROI
    span = None  # None is the full scan
    x0 = (0, 210)  # The location of the reference within the ROI
    # Set the image processing parameters
    sigma = 20
    equalize = True
    truncate = True
    # Set the small strain flag
    small_strain = False
    # Set the stiffness tensor
    C = utilities.get_stiffness_tensor(165.64, 63.94, 79.51, structure="cubic")
    # Calculate or read
    calc = False
    # Number of cores, max iterations, and convergence tolerance if calculating
    n_cores = 15
    max_iter = 100
    conv_tol = 1e-3
    ############################

    # Load the pattern object
    pat_obj, ang_data = utilities.get_scan_data(up2, ang)

    # Create the optimizer
    optimizer = get_homography.ICGNOptimizer(
        pat_obj=pat_obj,
        x0=x0,
        PC=ang_data.pc,
        sample_tilt=sample_tilt,
        detector_tilt=detector_tilt,
        pixel_size=pixel_size,
        step_size=step_size,
        scan_shape=ang_data.shape,
        small_strain=small_strain,
        C=C
    )
    # Set the image processing parameters
    optimizer.set_image_processing_kwargs(
        sigma=sigma, equalize=equalize, truncate=truncate
    )
    # Set the region of interest
    optimizer.set_roi(start=start, span=span)
    # Set the homography subset
    optimizer.set_homography_subset(subset_size, "image")
    # Set the initial guess parameters
    optimizer.set_initial_guess_params(
        subset_size=initial_guess_subset_size, init_type="full"
    )
    optimizer.print_setup()
    time.sleep(1)
    if calc:
        # Run the optimizer
        optimizer.run(
            n_cores=n_cores, max_iter=max_iter, conv_tol=conv_tol, verbose=False
        )
        optimizer.save(f"results/{name}_optimizer.pkl")
        # optimizer.save_results(f"results/{name}.pkl")
    else:
        optimizer.load(f"results/{name}_optimizer.pkl")
        # optimizer.load_results("results/CoNi90_DED_ICGN.pkl")
    e = optimizer.results.e
    

    e11 = e[0, :, 0, 0]
    e12 = e[0, :, 0, 1]
    e13 = e[0, :, 0, 2]
    e22 = e[0, :, 1, 1]
    e23 = e[0, :, 1, 2]
    e33 = e[0, :, 2, 2]
    e_tetra = (e11 + e22) / 2 - e33
    # e_tetra = e33 - (e11 + e22) / 2
    residuals = optimizer.results.residuals[0]
    num_iter = optimizer.results.num_iter[0]
    norms = optimizer.results.norms[0]
    x = np.arange(len(e11))

    mask = np.zeros(len(x), dtype=bool)

    color = np.array([(254, 188, 17) for _ in range(len(x))])
    color[mask] = (0, 54, 96)
    color = color / 255

    fig, ax = plt.subplots(3, 3, figsize=(15, 14))
    ax[0, 0].scatter(x, e11, c=color, marker="s", label=r"$\epsilon_{11}$")
    ax[0, 1].scatter(x, e12, c=color, marker="s", label=r"$\epsilon_{12}$")
    ax[0, 2].scatter(x, e13, c=color, marker="s", label=r"$\epsilon_{13}$")
    ax[1, 1].scatter(x, e22, c=color, marker="s", label=r"$\epsilon_{22}$")
    ax[1, 2].scatter(x, e23, c=color, marker="s", label=r"$\epsilon_{23}$")
    ax[2, 2].scatter(x, e33, c=color, marker="s", label=r"$\epsilon_{33}$")
    ax[1, 0].scatter(x, residuals, c=color, marker="s", label="Residuals")
    ax[2, 0].scatter(x, num_iter, c=color, marker="s", label="Num Iterations")
    ax[2, 1].scatter(x, e_tetra, c=color, marker="s", label=r"$\epsilon_{tetragonal}$")

    for a in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2], ax[2, 2], ax[2, 1]]:
        a.set_ylim(-0.022, 0.022)
        # a.set_ylim(-0.0018, 0.0018)
        utilities.standardize_axis(a)
        utilities.make_legend(a)

    for a in [ax[1, 0], ax[2, 0]]:#, ax[2, 1]]:
        utilities.standardize_axis(a)
        utilities.make_legend(a)

    plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.07, right=0.99, top=0.99, bottom=0.05)
    plt.savefig(f"E:/SiGe/{name}_Sample1.png", dpi=300)
