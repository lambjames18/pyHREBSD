import numpy as np
import timeit
import torch
from skimage import io
from scipy import linalg

if torch.cuda.is_available():
    device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')

def W_vectorized(p) -> np.ndarray:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    in_shape = p.shape[:-1]
    _0 = np.zeros(in_shape + (1,))
    return np.append(p, _0, axis=-1).reshape(in_shape + (3, 3,)) + np.eye(3)[None, ...]

def W(p) -> np.ndarray:
    """Convert homographies into a shape function.
    Assumes p is a (Nx8) array."""
    return np.append(p, 0).reshape(3, 3) + np.eye(3)

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

def main(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    c, lower = linalg.cho_factor(H)
    for i in range(t.shape[0]):
        for j in range(10):
            e = r - normalize(t[i])
            dC_IC_ZNSSD = 2 / r_zmsv * np.matmul(e, NablaR_dot_Jac.T)  # 8x1
            dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]
            norm = dp_norm(dp, xi)
            Wp = W(p[i])
            Wdp = W(dp)
            Wpdp = Wp.dot(np.linalg.inv(Wdp))
            p_out = ((Wpdp / Wpdp[2, 2]) - np.eye(3)).flatten()[:8]

def main_vectorized(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    for j in range(10):
        e = r - normalize(t)
        dC_IC_ZNSSD = 2 / r_zmsv * np.einsum('ij,kj->ik', e, NablaR_dot_Jac)  # 8x1
        c, lower = linalg.cho_factor(H)
        dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.T).T
        norm = dp_norm_vectorized(dp, xi)
        Wp = W_vectorized(p)
        Wdp = W_vectorized(dp)
        Wpdp = Wp @ np.linalg.inv(Wdp)
        p = ((Wpdp / Wpdp[:, 2, 2][:, None, None]) - np.eye(3)[None, ...])
        p = p.reshape(-1, 9)[:, :8]


path = "/Users/jameslamb/Library/Mobile Documents/com~apple~CloudDocs/Downloads/CoNi67_Pattern.tif"
r = io.imread(path)
t = r

x = np.arange(r.shape[1]) - 512
y = np.arange(r.shape[0]) - 512
X, Y = np.meshgrid(x, y)
xi = np.array([Y.flatten(), X.flatten()]).astype(np.float32)
r_zmsv = np.sqrt(((r - r.mean())**2).sum()).astype(np.float32)
NablaR_dot_Jac = np.random.rand(8, r.shape[0]*r.shape[1]).astype(np.float32)
H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)
p = np.random.rand(8).astype(np.float32)
r = r.flatten().astype(np.float32)
t = t.flatten().astype(np.float32)

t_v = np.repeat(r[None, :], 10, axis=0)
p_v = np.repeat(p[None, :], 10, axis=0)
p_out = main_vectorized(r, t_v, p_v, NablaR_dot_Jac, H, r_zmsv, xi)

t0 = timeit.timeit(lambda: main(r, t_v, p_v, NablaR_dot_Jac, H, r_zmsv, xi), number=100) / 1000
print(f"Time: {t0} seconds/solution")
t1 = timeit.timeit(lambda: main_vectorized(r, t_v, p_v, NablaR_dot_Jac, H, r_zmsv, xi), number=100) / 1000
print(f"Time: {t1} seconds/solution")
# dp_norm_speedup = 93.5
exit()

iterations = 10000

print("CPU")
tcpu = timeit.timeit(lambda: main_cpu(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi), number=iterations)
print(f"Time: {tcpu/iterations} seconds/iteration")
