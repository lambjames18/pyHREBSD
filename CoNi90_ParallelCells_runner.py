import numpy as np
import matplotlib.pyplot as plt

import utilities
import get_homography

if __name__ == "__main__":
    ############################
    # Load the pattern object
    up2 = "E:/cells/CoNi90-ParallelCells_2048x2048.up2"  # File doesnt exist at the moment
    up2 = "F:/CoNi90-ParallelCells_2048x2048.up2"
    ang = "E:/cells/CoNi90-ParallelCells.ang"
    name = "CoNi90-OrthoCells"
    # Set the geometry parameters
    pixel_size = 13.0
    sample_tilt = 70.0  # The sample tilt in degrees
    detector_tilt = 9.8  # The detector tilt in degrees
    subset_size = 1600
    initial_guess_subset_size = 1024
    # Set the roi parameters
    start = (0, 0)  # The pixel location to start the ROI
    span = (
        200,
        200,
    )  # The (row, column) span of the ROI from the start location (down, right)
    x0 = (100, 100)  # The location of the reference within the ROI
    # Set the image processing parameters
    sigma = 20
    equalize = True
    truncate = True
    # Set the small strain flag
    small_strain = False
    # Calculate or read
    calc = True
    # Number of cores, max iterations, and convergence tolerance if calculating
    n_cores = 20
    max_iter = 100
    conv_tol = 1e-3
    ############################
    
    # Load the pattern object
    pat_obj, ang_data = utilities.get_scan_data(up2, ang)
    pats = utilities.get_patterns(pat_obj, [0, 10000, 39999])
    print(pats.min(axis=(1, 2)), pats.max(axis=(1, 2)))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(pats[0], cmap="gray")
    ax[0].set_title("Pattern 0")
    ax[1].imshow(pats[1], cmap="gray")
    ax[1].set_title("Pattern 10000")
    ax[2].imshow(pats[2], cmap="gray")
    ax[2].set_title("Pattern 39999")
    plt.show()

    # Create the optimizer
    optimizer = get_homography.ICGNOptimizer(
        pat_obj,
        x0,
        ang_data.pc,
        sample_tilt,
        detector_tilt,
        ang_data.shape,
        small_strain,
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
    if calc:
        # Run the optimizer
        optimizer.run(n_cores=n_cores, max_iter=max_iter, conv_tol=conv_tol, verbose=False)
        optimizer.save_results(f"results/{name}.pkl")
    else:
        optimizer.load_results("results/CoNi90_DED_ICGN.pkl")
    # Save the results
    r = optimizer.results
    m = optimizer.roi

    # Generate maps
    utilities.view_tensor_images(r.F[m].reshape(span + (3, 3)), "jet", "deformation", (x0[0] - start[0], x0[1] - start[1]), "results", name)
    utilities.view_tensor_images(r.e[m].reshape(span + (3, 3)), "jet", "strain", (x0[0] - start[0], x0[1] - start[1]), "results", name, "upper")
    utilities.view_tensor_images(r.w[m].reshape(span + (3, 3)), "jet", "strain", (x0[0] - start[0], x0[1] - start[1]), "results", name, "upper")
    utilities.view_tensor_images(r.homographies[m].reshape(span + (8,)), "jet", "homography", (x0[0] - start[0], x0[1] - start[1]), "results", name)
    plt.close("all")

    # Save the ICGN optimization results (for logging/debugging purposes)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im0 = ax[0].imshow(r.num_iter[m].reshape(span), cmap="viridis")
    ax[0].set_title("Iteration count")
    im1 = ax[1].imshow(r.residuals[m].reshape(span), cmap="viridis")
    ax[1].set_title("Residuals")
    im2 = ax[2].imshow(r.norms[m].reshape(span), cmap="viridis")
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
            r.w[..., 2, 1] - r.w[..., 1, 2],
            r.w[..., 0, 2] - r.w[..., 2, 0],
            r.w[..., 1, 0] - r.w[..., 0, 1],
        ]
    )
    mask = u[2] < 0
    theta = np.arcsin(np.linalg.norm(u, axis=0) / 2) * 180 / np.pi
    theta[mask] *= -1
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(theta[m].reshape(span), cmap="viridis")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.01)
    l = ax.get_position()
    cax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    plt.colorbar(im, cax=cax)
    plt.savefig(f"results/{name}_rotation.png")
    plt.close(fig)
