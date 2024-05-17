import time
import utilities
import pyHREBSD
import numpy as np

if __name__ == "__main__":
    # up2 = "E:/cells/CoNi90-OrthoCells.up2"
    # ang = "E:/cells/CoNi90-OrthoCells.ang"
    # save_name = f"CoNi90-OrthoCells"
    up2 = "E:/GaN/20240425_27146.up2"
    ang = "E:/GaN/20240425_27146.ang"
    save_name = f"GaN"
    C = utilities.get_stiffness_tensor(237.0e3, 157.0e3, 140.0e3, structure="cubic")
    pixel_size = 13.0
    Nxy = (256, 256)
    b = 8
    x0 = (0, 0)
    size = 40
    DoG_sigmas = (1.0, 10.0)

    t0 = time.time()

    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang, Nxy, pixel_size, b)
    PC = ang_data.pc
    print(PC)
    print(pat_obj.patshape)
    exit()

    # # Get patterns
    # # idx = utilities.get_index(point, size, ang_data)
    # # pats = utilities.get_patterns(pat_obj)
    # pats = utilities.get_patterns(pat_obj)
    # pats = pats.reshape(ang_data.shape + pats.shape[-2:])
    # print(pats.shape)

    # # Process patterns
    # pats = utilities.process_patterns(pats, equalize=True, dog_sigmas=DoG_sigmas)

    # # Get initial guesses
    # R = pats[x0]
    # p0 = pyHREBSD.get_initial_guess(R, pats, PC)

    # # Get homographies
    # subset_slice = (slice(10, -10), slice(10, -10))
    # p = pyHREBSD.get_homography(R, pats, subset_slice=subset_slice, p0=p0, max_iter=50, conv_tol=1e-4, PC=None)
    # np.save(f"{save_name}_p.npy", p)
    p = np.load(f"{save_name}_p.npy")

    # Get deformation gradients and strain
    Fe = pyHREBSD.homography_to_elastic_deformation(p, PC)
    e, w = pyHREBSD.deformation_to_stress_strain(Fe)
    # s, e, w = deformation_to_stress_strain(Fe, C)

    # Print time
    t1 = time.time()
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f">Total time for the entire run: {hours:.0f}:{minutes:.0f}:{seconds:.0f}")

    # Save the results
    xy = x0 # The location of the reference point
    utilities.view_tensor_images(Fe, tensor_type="deformation", xy=xy, save_name=save_name, save_dir="results/")
    utilities.view_tensor_images(e, tensor_type="strain", xy=xy, save_name=save_name, save_dir="results/", show="upper")
    utilities.view_tensor_images(w, tensor_type="rotation", xy=xy, save_name=save_name, save_dir="results/", show="upper")
    # utilities.view_tensor_images(s, tensor_type="stress", xy=xy, save_name=save_name, save_dir="results/", show="upper")
    utilities.view_tensor_images(p, tensor_type="homography", xy=xy, save_name=save_name, save_dir="results/", show="all")