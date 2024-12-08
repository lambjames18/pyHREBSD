import time
import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities
import get_homography_cpu as gh_cpu
import get_homography_gpu as gh_gpu

if __name__ == "__main__":
    ############################
    # Load the pattern object
    sample = "A"  # The sample to analyze, "A" or "B"
    name = dict(A="ScanA", B="ScanB")[sample]
    up2 = dict(A="E:/SiGe/a-C03-scan/ScanA_1024x1024.up2", B="E:/SiGe/b-C04-scan/ScanB_1024x1024.up2")[sample]
    ang = dict(A="E:/SiGe/a-C03-scan/ScanA.ang", B="E:/SiGe/b-C04-scan/ScanB.ang")[sample]
    # Set the geometry parameters
    pixel_size = 26.0  # The pixel size in um, taking binning into account (so 4xpixel_size for 4x4 binning)
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 10.9  # The detector tilt in degrees
    step_size = 0.450  # The step size in um
    subset_size = 819
    fixed_projection = False
    # Set the initial guess parameters
    init_type = "none"  # The type of initial guess to use, "none", "full", or "partial"
    initial_guess_subset_size = 1024
    # Set the roi parameters
    start = (0, 0)  # The pixel location to start the ROI
    span = None  # None is the full scan
    x0 = (0, 0)  # The location of the reference within the ROI
    # Set the image processing parameters
    high_pass_sigma = 101
    low_pass_sigma = 2.5
    truncate_std_scale = 3.0
    # Set the small strain flag
    small_strain = False
    # Set the stiffness tensor
    # C = utilities.get_stiffness_tensor(165.77, 63.94, 79.62, structure="cubic")
    C = utilities.get_stiffness_tensor(158.0, 61.0, 78, structure="cubic")
    traction_free = True
    # Calculate or read
    calc = False
    # Whether to view the reference image
    view_reference = False
    # Number of cores, max iterations, and convergence tolerance if calculating
    n_cores = 10
    max_iter = 50
    conv_tol = 1e-3
    # Verbose
    verbose = False
    gpu = False
    ############################

    if gpu:
        name += "_gpu"
        get_homography = gh_gpu
    else:
        # name += "_cpu"
        get_homography = gh_cpu

    # Load the pattern object
    # pat_obj, ang_data = utilities.get_scan_data(up2, ang)
    pat_obj = Data.UP2(up2)
    ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
    # Rotate the stiffness tensor into the sample frame
    if C is not None:
        C = utilities.rotate_stiffness_to_sample_frame(C, ang_data.quats)
        # C = np.ones(ang_data.shape + (6, 6), dtype=float) * C
    # Correct PC
    PC = np.array([ang_data.pc[0] - ang_data.shape[1] / 2, ang_data.pc[1] - ang_data.shape[0] / 2, ang_data.pc[2]])
    if calc:
        # Create the optimizer
        optimizer = get_homography.ICGNOptimizer(
            pat_obj=pat_obj,
            x0=x0,
            PC=PC,
            sample_tilt=sample_tilt,
            detector_tilt=detector_tilt,
            pixel_size=pixel_size,
            step_size=step_size,
            scan_shape=ang_data.shape,
            small_strain=small_strain,
            C=C,
            fixed_projection=fixed_projection,
            traction_free=traction_free,
        )
        
        # Set the image processing parameters
        optimizer.set_image_processing_kwargs(
            low_pass_sigma=low_pass_sigma,
            high_pass_sigma=high_pass_sigma,
            truncate_std_scale=truncate_std_scale
            
        )
        # Set the region of interest
        optimizer.set_roi(start=start, span=span)
        # Set the homography subset
        optimizer.set_homography_subset(subset_size, "image")
        # Set the initial guess parameters
        optimizer.set_initial_guess_params(
            subset_size=initial_guess_subset_size, init_type=init_type
        )
        if view_reference:
            optimizer.view_reference()
            time.sleep(1)

        # Run the optimizer
        # optimizer.extra_verbose = True
        # optimizer.run(
        #     n_cores=n_cores, max_iter=max_iter, conv_tol=conv_tol, verbose=verbose
        # )
        optimizer.run(
            batch_size=16, max_iter=max_iter, conv_tol=conv_tol
        )
        results = optimizer.results
        results.save(f"results/{name}_results.pkl")
        results.calculate()
        results.save(f"results/{name}_results.pkl")
    else:
        results = get_homography.Results(
            ang_data.shape,
            PC,
            x0,
            step_size / pixel_size,
            fixed_projection,
            detector_tilt,
            sample_tilt,
            traction_free,
            small_strain,
            C,
        )
        results.load(f"results/{name}_results.pkl")
        results.shape = ang_data.shape
        results.C = C
        results.PC_array = np.ones(ang_data.shape + (3,), dtype=float) * PC
        results.traction_free = traction_free
        results.small_strain = small_strain
        results.calculate()
    h23 = results.homographies[0, :, 5]
    h22 = results.homographies[0, :, 4]
    h13 = results.homographies[0, :, 2]
    e = results.strains
    e11 = e[0, :, 0, 0]
    e12 = e[0, :, 0, 1]
    e13 = e[0, :, 0, 2]
    e22 = e[0, :, 1, 1]
    e23 = e[0, :, 1, 2]
    e33 = e[0, :, 2, 2]
    e_t = e33 - (e11 + e22) / 2
    w21 = -np.rad2deg(results.rotations[0, :, 0, 1])
    w13 = np.rad2deg(results.rotations[0, :, 0, 2])
    w32 = -np.rad2deg(results.rotations[0, :, 1, 2])
    res = results.residuals[0]
    itr = results.num_iter[0]
    x = np.arange(len(e11))

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].plot(x, e11, lw=3, c="r", label=r"$\epsilon_{11}$")
    ax[0, 0].plot(x, e22, lw=3, c="g", label=r"$\epsilon_{22}$")
    ax[0, 0].plot(x, e33, lw=3, c="b", label=r"$\epsilon_{33}$")
    ax[0, 1].plot(x, e12, lw=3, c="m", label=r"$\epsilon_{12}$")
    ax[0, 1].plot(x, e13, lw=3, c="y", label=r"$\epsilon_{13}$")
    ax[0, 1].plot(x, e23, lw=3, c="c", label=r"$\epsilon_{23}$")
    ax[0, 2].plot(x, e_t, lw=3, c="k", label=r"$\epsilon_{tetragonal}$")
    ax[1, 0].plot(x, res, lw=3, c="k", label="Residuals")
    ax[1, 1].plot(x, itr, lw=3, c="k", label="Num Iterations")
    # ax[1, 2].plot(x, w13, lw=3, c="tab:orange", label=r"$\omega_{13}$")
    # ax[1, 2].plot(x, w21, lw=3, c="tab:purple", label=r"$\omega_{21}$")
    # ax[1, 2].plot(x, w32, lw=3, c="tab:brown", label=r"$\omega_{32}$")
    ax[1, 2].plot(x, h23, lw=3, c="tab:orange", label=r"$h_{23}$")
    ax[1, 2].plot(x, h22, lw=3, c="tab:purple", label=r"$h_{22}$")
    ax[1, 2].plot(x, h13, lw=3, c="tab:brown", label=r"$h_{13}$")

    bound = 0.02
    for a in [ax[0, 0], ax[0, 1], ax[0, 2]]:
        a.set_ylim(-bound, bound)

    args = [dict(ncols=3), dict(ncols=3), dict(ncols=1), dict(ncols=1), dict(ncols=1), dict(ncols=3)]
    for i, a in enumerate(ax.flatten()):
        utilities.standardize_axis(a)
        utilities.make_legend(a, columnspacing=0.8, handlelength=1, **args[i])

    plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.08, right=0.99, top=0.95, bottom=0.05)
    plt.savefig(f"E:/SiGe/{name}_Python-results.png", dpi=300)
    # plt.savefig(f"E:/SiGe/{name}_Python-results_corrected.png", dpi=300)
    print("Max", e_t.max())
    print("Max mean", e_t[e_t > 0.005].mean())
