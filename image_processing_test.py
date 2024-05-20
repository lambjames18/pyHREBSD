import numpy as np
import matplotlib.pyplot as plt

import utilities
import pyHREBSD


if __name__ == "__main__":
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

    ### Parameters ###


    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang)

    idx = np.arange(0, pat_obj.nPatterns)
    idx = [idx[0], idx[400]]

    # Set the homography center properly
    if h_center == "pattern":
        PC = ang_data.pc
    elif h_center == "image":
        PC = (pat_obj.patshape[1] / 2, pat_obj.patshape[0] / 2, ang_data.pc[2])

    # Get patterns
    pats = utilities.get_patterns(pat_obj, idx=idx).astype(float)
    pats = utilities.process_patterns(pats, sigma=sigma, equalize=equalize,truncate=truncate)

    # Get initial guesses
    R = pats[0]
    T = pats[1]
    tilt = 90 - sample_tilt + detector_tilt
    p0 = pyHREBSD.get_initial_guess(R, T, PC, tilt, initial_subset_size, guess_type)
    print("Init p:  {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e}".format(*p0))

    # Get homographies
    row_s = slice(max(int(PC[1] - subset_size[0] / 2), 0),
                  min(int(PC[1] + subset_size[0] / 2), pat_obj.patshape[0]))
    col_s = slice(max(int(PC[0] - subset_size[1] / 2), 0),
                  min(int(PC[0] + subset_size[1] / 2), pat_obj.patshape[1]))
    subset_slice = (row_s, col_s)
    print(subset_slice)
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
    print("Number of iterations:", i_count)
    print("Final p: {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e} {:.5e}".format(*p))

    # Get deformation gradients and strain
    PC_mod = (ang_data.pc[0] - 1024, ang_data.pc[1] - 1024, PC[2])
    Fe = pyHREBSD.homography_to_elastic_deformation(p, PC_mod)
    C = utilities.get_stiffness_tensor(165.6, 63.9, 79.5, structure="cubic")
    e, w, s = pyHREBSD.deformation_to_stress_strain(Fe, C, small_strain=False)
    e11, e12, e13, e21, e22, e23, e31, e23, e33 = e.flatten()
    print("Strain: e11={:.5e}, e22={:.5e}, e33={:.5e}, e12={:.5e} e23={:.5e}, e13={:.5e}".format(e11, e22, e33, e12, e23, e13))


    R_0 = pyHREBSD.normalize(pyHREBSD.deform_image(R, np.zeros(8), ang_data.pc, subset_slice))
    T_0 = pyHREBSD.normalize(pyHREBSD.deform_image(T, np.zeros(8), ang_data.pc, subset_slice))
    T_1 = pyHREBSD.normalize(pyHREBSD.deform_image(T, p, ang_data.pc, subset_slice))

    e_0 = R_0 - T_0
    e_1 = R_0 - T_1
    fig2, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    ax[0, 0].imshow(R_0, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Reference')
    ax[0, 1].imshow(e_0, cmap='gray', vmin=-1e-3, vmax=1e-3)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Original Difference')
    ax[1, 0].imshow(T_1, cmap='gray')
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Final guess')
    ax[1, 1].imshow(e_1, cmap='gray', vmin=-1e-3, vmax=1e-3)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Final Difference')
    fig2.suptitle('Comparison of the reference and final guess')
    plt.tight_layout()
    plt.show()
