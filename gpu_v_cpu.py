import numpy as np
import timeit
import torch
from skimage import io
from scipy import linalg
import mpire

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Vectorized CPU implementation

def W_vectorized(p) -> np.ndarray:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    in_shape = p.shape[:-1]
    _0 = np.zeros(in_shape + (1,))
    return np.append(p, _0, axis=-1).reshape(in_shape + (3, 3,)) + np.eye(3)[None, ...]


def normalize_vectorized(img):
    """Zero mean, unit variance normalization of an image.
    Assumes the images has been flattened to a 2D array (N, M),
    where N is the number of images and M is the number of pixels."""
    img_bar = img.mean(axis=-1)[...,None]#, keepdims=True)
    dimg_tilde = np.sqrt(((img - img_bar)**2).sum(axis=-1)[..., None])#, keepdims=True))
    return (img - img_bar) / dimg_tilde


def dp_norm_vectorized(dp, xi) -> float:
    """Compute the norm of the delta p vector.
    Assumes dp is a (Nx8) array and xi is a (2,M) array."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([[xi1max, xi2max]])
    dp_i0 = np.square(dp[:, 0:2] * ximax).sum(axis=-1)
    dp_i1 = np.square(dp[:, 3:5] * ximax).sum(axis=-1)
    dp_i2 = np.square(dp[:, 6:8] * ximax).sum(axis=-1)
    out = np.sqrt(dp_i0 + dp_i1 + dp_i2 + np.square(dp[:, 2]) + np.square(dp[:, 5]))
    return out

# Non-vectorized CPU implementation

def W(p) -> np.ndarray:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    return np.append(p, 0).reshape(3, 3) + np.eye(3)


def normalize(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    img_bar = img.mean()
    dimg_tilde = np.sqrt(((img - img_bar)**2).sum())
    out = (img - img_bar) / dimg_tilde
    return out


def dp_norm(dp, xi) -> float:
    """Compute the norm of the deformation increment.
    This is essentially a modified form of a homography magnitude.

    Args:
        dp (np.ndarray): The deformation increment. Shape is (8,).
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        float: The norm of the deformation increment."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([xi1max, xi2max])
    dp_i0 = np.array([dp[0], dp[1]]) * ximax
    dp_i1 = np.array([dp[3], dp[4]]) * ximax
    dp_i2 = np.array([dp[6], dp[7]]) * ximax
    out = np.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
    return out

# Vectorized GPU implementation

def W_vectorized_gpu(p) -> torch.Tensor:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    in_shape = p.shape[:-1]
    _0 = torch.zeros(in_shape + (1,), device=device)
    return torch.cat((p, _0), dim=-1).reshape(in_shape + (3, 3,)) + torch.eye(3, device=device)[None, ...]


def normalize_vectorized_gpu(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    img_bar = img.mean(axis=-1)[..., None]
    dimg_tilde = torch.sqrt(((img - img_bar)**2).sum(axis=-1)[..., None])
    return (img - img_bar) / dimg_tilde


def dp_norm_vectorized_gpu(dp, xi) -> float:
    """Compute the norm of the delta p vector.
    Assumes dp is a (Nx8) array and xi is a (2,M) array."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = torch.tensor([[xi1max, xi2max]], device=device)
    dp_i0 = torch.square(dp[:, 0:2] * ximax).sum(axis=-1)
    dp_i1 = torch.square(dp[:, 3:5] * ximax).sum(axis=-1)
    dp_i2 = torch.square(dp[:, 6:8] * ximax).sum(axis=-1)
    out = torch.sqrt(dp_i0 + dp_i1 + dp_i2 + torch.square(dp[:, 2]) + torch.square(dp[:, 5]))
    return out

# Non-vectorized GPU implementation

def W_gpu(p) -> torch.Tensor:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    return torch.cat((p, torch.zeros(1, device=device)), dim=-1).reshape(3, 3) + torch.eye(3, device=device)

def normalize_gpu(img):
    """Zero-mean normalize an image with unit standard deviation.
    Note that the standard deviation is multiplied by the number of pixels minus one.
    Args:
        img (np.ndarray): The image to normalize.
    Returns:
        np.ndarray: The normalized image."""
    img_bar = img.mean()
    dimg_tilde = torch.sqrt(((img - img_bar)**2).sum())
    out = (img - img_bar) / dimg_tilde
    return out

def dp_norm_gpu(dp, xi) -> float:
    """Compute the norm of the deformation increment.
    This is essentially a modified form of a homography magnitude.

    Args:
        dp (np.ndarray): The deformation increment. Shape is (8,).
        xi (np.ndarray): The subset coordinates. Shape is (2, N).

    Returns:
        float: The norm of the deformation increment."""
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = torch.tensor([xi1max, xi2max], device=device)
    dp_i0 = dp[0:2] * ximax
    dp_i1 = dp[3:5] * ximax
    dp_i2 = dp[6:8] * ximax
    out = torch.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
    return out

# Main functions

def main_cpu(*args):
    r, t, p, NablaR_dot_Jac, H, r_zmsv, xi = args
    for j in range(20):
        e = r - normalize(t)
        dC_IC_ZNSSD = 2 / r_zmsv * np.matmul(e, NablaR_dot_Jac.T)  # 8x1
        c, lower = linalg.cho_factor(H)
        dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]
        norm = dp_norm(dp, xi)
        Wp = W(p)
        Wdp = W(dp)
        Wpdp = Wp.dot(np.linalg.inv(Wdp))
        p = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).flatten()[:8]


def main_gpu(*args):
    r, t, p, NablaR_dot_Jac, L, r_zmsv, xi = args
    for j in range(20):
        e = r - normalize_gpu(t)
        dC_IC_ZNSSD = 2 / r_zmsv * torch.matmul(e, NablaR_dot_Jac.T)  # 8x1
        dp = torch.cholesky_solve(-dC_IC_ZNSSD.reshape(-1, 1), L)[:, 0]
        norm = dp_norm_gpu(dp, xi)
        Wp = W_gpu(p)
        Wdp = W_gpu(dp)
        Wpdp = torch.matmul(Wp, torch.linalg.inv(Wdp))
        p = ((Wpdp / Wpdp[2, 2]) - torch.eye(3, device=device)).reshape(9)[:8]


def main(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    # c, lower = linalg.cho_factor(H)
    args = [(r, t[i], p[i], NablaR_dot_Jac, H, r_zmsv, xi) for i in range(t.shape[0])]
    with mpire.WorkerPool(n_jobs=10) as pool:
        pool.map(main_cpu, args)


def main_vectorized(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    for j in range(20):
        e = r - normalize_vectorized(t)
        dC_IC_ZNSSD = 2 / r_zmsv * np.einsum('ij,kj->ik', e, NablaR_dot_Jac)  # 8x1
        c, lower = linalg.cho_factor(H)
        dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.T).T
        norm = dp_norm_vectorized(dp, xi)
        Wp = W_vectorized(p)
        Wdp = W_vectorized(dp)
        Wpdp = Wp @ np.linalg.inv(Wdp)
        p = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - np.eye(3)[None, ...])
        p = p.reshape(-1, 9)[:, :8]


def main_g(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    r = torch.tensor(r, device=device)
    t = torch.tensor(t, device=device)
    p = torch.tensor(p, device=device)
    NablaR_dot_Jac = torch.tensor(NablaR_dot_Jac, device=device)
    H = torch.tensor(H, device=device)
    L = torch.linalg.cholesky(H)
    xi = torch.tensor(xi, device=device)
    for i in range(t.shape[0]):
        main_gpu(r, t[i], p[i], NablaR_dot_Jac, L, r_zmsv, xi)


def main_vectorized_gpu(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    r = torch.tensor(r, device=device)
    t = torch.tensor(t, device=device)
    p = torch.tensor(p, device=device)
    NablaR_dot_Jac = torch.tensor(NablaR_dot_Jac, device=device)
    H = torch.tensor(H, device=device)
    L = torch.linalg.cholesky(H)
    xi = torch.tensor(xi, device=device)
    for j in range(20):
        e = r - normalize_vectorized_gpu(t)
        dC_IC_ZNSSD = 2 / r_zmsv * torch.einsum('ij,kj->ik', e, NablaR_dot_Jac)  # 8x1
        dp = torch.cholesky_solve(-dC_IC_ZNSSD.T, L).T
        norm = dp_norm_vectorized_gpu(dp, xi)
        Wp = W_vectorized_gpu(p)
        Wdp = W_vectorized_gpu(dp)
        Wpdp = Wp @ torch.linalg.inv(Wdp)
        p = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - torch.eye(3, device=device)[None, ...])
        p = p.reshape(-1, 9)[:, :8]


if __name__ == "__main__":
    import utilities
    import pyHREBSD
    import pyHREBSD_GPU
    import time
    from tqdm.auto import tqdm
    ### Parameters ###
    # Names and paths
    sample = "A"  # The sample to analyze, "A" or "B"
    save_name = dict(A=f"SiGeScanA", B=f"SiGeScanB")[sample]
    up2 = dict(A="E:/SiGe/ScanA.up2", B="E:/SiGe/ScanB.up2")[sample]
    ang = dict(A="E:/SiGe/ScanA.ang", B="E:/SiGe/ScanB.ang")[sample]

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

    ### Parameters ###

    # Read in data
    pat_obj, ang_data = utilities.get_scan_data(up2, ang)

    idx = np.arange(80, 90)

    # Set the homography center properly
    if h_center == "pattern":
        PC = ang_data.pc
    elif h_center == "image":
        PC = (pat_obj.patshape[1] / 2, pat_obj.patshape[0] / 2, ang_data.pc[2])

    # Get patterns
    print("Getting patterns...")
    pats = utilities.get_patterns(pat_obj, idx=idx).astype(float)
    pats = utilities.process_patterns(pats, sigma=sigma, equalize=equalize,truncate=truncate)

    # Get initial guesses
    print("Getting initial guess...")
    R = pats[x0]
    T = pats
    tilt = 90 - sample_tilt + detector_tilt
    p0 = pyHREBSD.get_initial_guess(R, T, PC, tilt, initial_subset_size, guess_type)

    # Do precomputations
    print("Precomputing...")
    subset_slice = (slice(int(PC[1] - subset_size[0] / 2), int(PC[1] + subset_size[0] / 2)),
                    slice(int(PC[0] - subset_size[1] / 2), int(PC[0] + subset_size[1] / 2)))
    r, r_zmsv, NablaR_dot_Jac, H, xi = pyHREBSD.reference_precompute(R, subset_slice, PC)

    # Parallel GPU
    t0 = time.time()
    p_vals_gpu_v = pyHREBSD_GPU.IC_GN_vectorized(p0, r, T, r_zmsv, NablaR_dot_Jac, H, xi, PC, conv_tol=conv_tol, max_iter=max_iter)
    t_gpu_v = (time.time() - t0)
    print(f"\tGPU vectorized time: {t_gpu_v} seconds ({t_gpu_v / T.shape[0]}s per pattern)")

    # print("Starting GPU timing test...")
    # p_vals_gpu = np.zeros_like(p0)
    # t0 = time.time()
    # for i in tqdm(range(T.shape[0])):
    #     p_vals_gpu[i] = pyHREBSD_GPU.IC_GN(p0[i], r, T[i], r_zmsv, NablaR_dot_Jac, H, xi, PC, conv_tol=conv_tol, max_iter=max_iter)
    # t_gpu = (time.time() - t0)
    # print(f"\tGPU time: {t_gpu} seconds (f{t_gpu / T.shape[0]}s per pattern)")

    print("Starting CPU timing test...")
    p_vals_cpu = np.zeros_like(p0)
    t0 = time.time()
    with mpire.WorkerPool(n_jobs=10) as pool:
        p_vals_cpu = pool.map(pyHREBSD.IC_GN, [(p0[i], r, T[i], r_zmsv, NablaR_dot_Jac, H, xi, PC, conv_tol, max_iter) for i in range(T.shape[0])])
    p_vals_cpu = p_vals_cpu.reshape(-1, 8)
    t_cpu = (time.time() - t0)
    print(f"\tCPU time: {t_cpu} seconds ({t_cpu / T.shape[0]}s per pattern)")

    # same = np.allclose(p_vals_cpu, p_vals_gpu, atol=1e-5, rtol=1e-5)
    # print("CPU and GPU outputs are the same?", same)
    # if not same:
    #     print(np.array(p_vals_cpu) - np.array(p_vals_gpu))
    same = np.allclose(p_vals_cpu, p_vals_gpu_v, atol=1e-5, rtol=1e-5)
    print("CPU and GPU vectorized outputs are the same?", same)
    if not same:
        print(zip(p_vals_cpu, p_vals_gpu_v))
