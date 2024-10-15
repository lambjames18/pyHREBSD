import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import Data
import utilities

if __name__ == "__main__":
    ############################
    # Load the pattern object
    name = "GaN27238"
    up2 = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_512x512_flipX.up2"
    ang = "/Users/jameslamb/Documents/research/data/GaN-DED/20240508_27238_flipX.ang"
    # Set the geometry parameters
    pixel_size = 26.0  # The pixel size in um, taking binning into account (so 4xpixel_size for 4x4 binning)
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 8.5  # The detector tilt in degrees
    step_size = 0.02  # The step size in um
    subset_size = 480
    fixed_projection = False
    # Set the initial guess parameters
    init_type = "none"  # The type of initial guess to use, "none", "full", or "partial"
    initial_guess_subset_size = 512
    # Set the roi parameters
    start = (0, 0)  # The pixel location to start the ROI
    span = None  # None is the full scan
    x0 = (100, 100)  # The location of the reference within the ROI
    # Set the image processing parameters
    high_pass_sigma = 75
    low_pass_sigma = 2.5
    truncate_std_scale = 3.0
    # Set the small strain flag
    small_strain = False
    # Set the stiffness tensor
    C = utilities.get_stiffness_tensor(365.0, 135.0, 114.0, 381.0, 109.0, structure="hexagonal")
    traction_free = True
    # Calculate or read
    calc = False
    # Whether to view the reference image
    view_reference = True
    # Max iterations, and convergence tolerance if calculating
    max_iter = 50
    conv_tol = 1e-3
    # Verbose
    verbose = False
    # If using the GPU, set the batch size, if CPU, set the number of cores
    gpu = False
    batch_size = 64
    n_cores = int(os.cpu_count() * 0.75)
    print(f"Using {n_cores} cores")
    ############################

    if gpu:
        name += "_gpu"
        import get_homography_gpu as get_homography
    else:
        # name += "_cpu"
        import get_homography as get_homography

    # Load the pattern object
    # pat_obj, ang_data = utilities.get_scan_data(up2, ang)
    pat_obj = Data.UP2(up2)
    ang_data = utilities.read_ang(ang, pat_obj.patshape, segment_grain_threshold=None)
    if span is None:
        span = ang_data.shape
    print("SCAN DATA LOADED")
    print("\tShape:", ang_data.shape)
    print("\tPC:", ang_data.pc)
    print("\tStep size:", step_size)
    print("\tPattern size:", pat_obj.patshape)
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
        if gpu:
            optimizer.run(
                batch_size=batch_size, max_iter=max_iter, conv_tol=conv_tol, verbose=verbose
            )
        else:
            optimizer.run(
                n_cores=n_cores, max_iter=max_iter, conv_tol=conv_tol, verbose=verbose
            )
        results = optimizer.results
        results.save(f"results/{name}_results.pkl")
        results.calculate()
        results.save(f"results/{name}_results.pkl")
    else:
        results = utilities.Results(
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
        results.calculate(free_to_dilate=False)

    m = results.num_iter > 0
    # Generate maps
    utilities.view_tensor_images(results.F[m].reshape(span + (3, 3)), "jet", "deformation", (x0[0] - start[0], x0[1] - start[1]), "results", name)
    utilities.view_tensor_images(results.strains[m].reshape(span + (3, 3)), "jet", "strain", (x0[0] - start[0], x0[1] - start[1]), "results", name, "upper")
    utilities.view_tensor_images(results.rotations[m].reshape(span + (3, 3)), "jet", "strain", (x0[0] - start[0], x0[1] - start[1]), "results", name, "upper")
    utilities.view_tensor_images(results.homographies[m].reshape(span + (8,)), "jet", "homography", (x0[0] - start[0], x0[1] - start[1]), "results", name)
    plt.close("all")

    # Save the ICGN optimization results (for logging/debugging purposes)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im0 = ax[0].imshow(results.num_iter[m].reshape(span), cmap="viridis")
    ax[0].set_title("Iteration count")
    im1 = ax[1].imshow(results.residuals[m].reshape(span), cmap="viridis")
    ax[1].set_title("Residuals")
    im2 = ax[2].imshow(results.norms[m].reshape(span), cmap="viridis")
    ax[2].set_title("Norms")
    plt.subplots_adjust(wspace=0.5, left=0.1, right=0.9, top=0.99, bottom=0.01)
    l = ax[0].get_position()
    cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    plt.colorbar(im0, cax=cax)
    l = ax[1].get_position()
    cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    plt.colorbar(im1, cax=cax)
    l = ax[2].get_position()
    cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    plt.colorbar(im2, cax=cax)
    plt.savefig(f"results/{name}_ICGN.png")
    plt.close(fig)

    # Save an alternate version of the rotation map
    u = np.array(
        [
            results.rotations[..., 2, 1] - results.rotations[..., 1, 2],
            results.rotations[..., 0, 2] - results.rotations[..., 2, 0],
            results.rotations[..., 1, 0] - results.rotations[..., 0, 1],
        ]
    )
    mask = u[2] < 0
    theta = np.arcsin(np.linalg.norm(u, axis=0) / 2) * 180 / np.pi
    theta[mask] *= -1
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(theta[m].reshape(span), cmap="viridis")
    plt.subplots_adjust(left=0.1, right=0.85, top=0.99, bottom=0.01)
    l = ax.get_position()
    cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    plt.colorbar(im, cax=cax)
    plt.savefig(f"results/{name}_rotation.png")
    plt.close(fig)
