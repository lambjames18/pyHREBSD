import time
import numpy as np
import matplotlib.pyplot as plt

import pyHREBSD
import utilities

if __name__ == "__main__":
    save_name = f"SiGeScanA"
    up2 = "E:/SiGe/ScanA.up2"
    pixel_size = 13.0  # The pixel size in um
    Nxy = (2048, 2048)  # The number of pixels in the x and y directions on the detector
    x0 = (0)  # The location of the reference point
    DoG_sigmas = (1.0, 20.0)  # The sigmas for the difference of Gaussians filter
    # PC = (-0.0703, 0.0747, 13312.0000)
    PC = (-0.0703, 0.0747, 1618.0)
    print(utilities.convert_pc(PC, Nxy, pixel_size, in_format="std", out_format="edax"))

    t0 = time.time()
    # Read in data
    pat_obj = utilities.read_up2(up2)
    # idx = np.array([0, 40, 100, 140, 180])
    idx = np.arange(60, 100, 1)
    pats = utilities.get_patterns(pat_obj, idx=idx)

    # Process patterns
    # pats_clean = utilities.process_patterns(pats, equalize=True, dog_sigmas=DoG_sigmas)
    pats = utilities.process_patterns(pats, equalize=True, dog_sigmas=DoG_sigmas)

    # Get initial guesses
    R = pats[x0]
    p0 = pyHREBSD.get_initial_guess(R, pats)

    # Get homographies
    subset_slice = (slice(10, -10), slice(10, -10))
    # subset_slice = (slice(None), slice(None))
    p = pyHREBSD.get_homography(R, pats, subset_slice=subset_slice, p0=p0, PC=PC, max_iter=50, conv_tol=1e-4, parallel=True)
    np.save(f"{save_name}_p.npy", p)

    # Get deformation gradients and strain
    Fe = pyHREBSD.homography_to_elastic_deformation(p, PC)
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
    # ax.axvline(81, color="black", linestyle="--")
    # ax.axvline(118, color="black", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Strain")
    ax.legend()
    plt.tight_layout()
    plt.show()
