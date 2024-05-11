import time
import numpy as np
import matplotlib.pyplot as plt

import pyHREBSD
import utilities

if __name__ == "__main__":
    t0 = time.time()

    save_name = f"SiGeScanA"
    up2 = "E:/SiGe/ScanA.up2"
    ang = "E:/SiGe/ScanA.ang"
    pixel_size = 13.0  # The pixel size in um
    Nxy = (2048, 2048)  # The number of pixels in the x and y directions on the detector
    x0 = (0)  # The location of the reference point
    DoG_sigmas = (1.0, 20.0)  # The sigmas for the difference of Gaussians filter
    calc = True

    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang, Nxy, pixel_size, 1)
    idx = np.arange(75, 95, 5)
    pats = utilities.get_patterns(pat_obj, idx=idx)
    pats[pats <= np.percentile(pats, 1)] = np.percentile(pats, 50)

    if calc:
        # Process patterns
        # pats_clean = utilities.process_patterns(pats, equalize=True, dog_sigmas=DoG_sigmas)
        pats = utilities.process_patterns(pats, equalize=True, dog_sigmas=DoG_sigmas)

        # Get initial guesses
        R = pats[x0]
        p0 = pyHREBSD.get_initial_guess(R, pats, ang_data.pc)
        print(p0)

        # Get homographies
        subset_slice = (slice(256, -256), slice(256, -256))
        # subset_slice = (slice(None), slice(None))
        p = pyHREBSD.get_homography(R, pats, subset_slice, p0=p0, PC=ang_data.pc, max_iter=50, conv_tol=1e-4, parallel=True)
        np.save(f"{save_name}_p.npy", p)

    else:
        p = np.load(f"{save_name}_p.npy")

    # Get deformation gradients and strain
    Fe = pyHREBSD.homography_to_elastic_deformation(p, ang_data.pc)
    e, w = pyHREBSD.deformation_to_stress_strain(Fe)

    # Print time
    t1 = time.time()
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f">Total time for the entire run: {hours:.0f}:{minutes:.0f}:{seconds:.0f}")

    # Save the results
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(idx, e[:, 0, 0], label="e_xx")
    ax.plot(idx, e[:, 1, 1], label="e_yy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Strain")
    ax.legend()
    plt.tight_layout()
    fig1, ax1 = plt.subplots(2, 4, figsize=(16, 8))
    ax1[0,0].plot(idx, p[:, 0])
    ax1[0,1].plot(idx, p[:, 1])
    ax1[0,2].plot(idx, p[:, 2])
    ax1[0,3].plot(idx, p[:, 3])
    ax1[1,0].plot(idx, p[:, 4])
    ax1[1,1].plot(idx, p[:, 5])
    ax1[1,2].plot(idx, p[:, 6])
    ax1[1,3].plot(idx, p[:, 7])
    ax1[0,0].set_title("p11")
    ax1[0,1].set_title("p12")
    ax1[0,2].set_title("p13")
    ax1[0,3].set_title("p21")
    ax1[1,0].set_title("p22")
    ax1[1,1].set_title("p23")
    ax1[1,2].set_title("p31")
    ax1[1,3].set_title("p32")
    plt.tight_layout()
    plt.show()
