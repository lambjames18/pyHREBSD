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

def W(p) -> np.ndarray:
    return np.array([[1 + p[0],     p[1], p[2]],
                     [    p[3], 1 + p[4], p[5]],
                     [    p[6],     p[7],    1]])

def W_gpu(p) -> torch.Tensor:
    return torch.tensor([[1 + p[0],     p[1], p[2]],
                         [    p[3], 1 + p[4], p[5]],
                         [    p[6],     p[7],    1]])

def normalize(img):
    img_bar = img.mean()
    dimg_tilde = np.sqrt(((img - img_bar)**2).sum())
    return (img - img_bar) / dimg_tilde

def normalize_gpu(img):
    img_bar = img.mean()
    dimg_tilde = torch.sqrt(((img - img_bar)**2).sum())
    return (img - img_bar) / dimg_tilde

def dp_norm(dp, xi) -> float:
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = np.array([xi1max, xi2max])
    dp_i0 = np.array([dp[0], dp[1]]) * ximax
    dp_i1 = np.array([dp[3], dp[4]]) * ximax
    dp_i2 = np.array([dp[6], dp[7]]) * ximax
    out = np.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
    return out

def dp_norm_gpu(dp, xi) -> torch.Tensor:
    xi1max = xi[0].max()
    xi2max = xi[1].max()
    ximax = torch.tensor([xi1max, xi2max])
    dp_i0 = torch.tensor([dp[0], dp[1]]) * ximax
    dp_i1 = torch.tensor([dp[3], dp[4]]) * ximax
    dp_i2 = torch.tensor([dp[6], dp[7]]) * ximax
    out = torch.sqrt((dp_i0**2).sum() + (dp_i1**2).sum() + (dp_i2**2).sum() + (dp[2]**2 + dp[5]**2))
    return out

def main_cpu(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    t = normalize(t)
    e = r - t
    dC_IC_ZNSSD = 2 / r_zmsv * np.matmul(e, NablaR_dot_Jac.T)  # 8x1
    c, lower = linalg.cho_factor(H)
    dp = linalg.cho_solve((c, lower), -dC_IC_ZNSSD.reshape(-1, 1))[:, 0]
    norm = dp_norm(dp, xi)
    Wp = W(p).dot(np.linalg.inv(W(dp)))
    Wp = (Wp / Wp[2, 2]) - np.eye(3)
    p = Wp.flatten()[:8]

def main_gpu(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi):
    t = torch.tensor(t).to(device)
    t = normalize_gpu(t)
    e = r - t
    dC_IC_ZNSSD = 2 / r_zmsv * torch.matmul(e, NablaR_dot_Jac.T)  # 8x1
    L = torch.linalg.cholesky(H)
    dp = torch.cholesky_solve(-dC_IC_ZNSSD.reshape(-1, 1), L)[:, 0]
    norm = dp_norm_gpu(dp, xi)
    Wp = W_gpu(p).matmul(torch.linalg.inv(W_gpu(dp)))
    Wp = (Wp / Wp[2, 2]) - torch.eye(3)
    p = Wp.flatten()[:8]

path = "/Users/jameslamb/Library/Mobile Documents/com~apple~CloudDocs/Downloads/CoNi67_Pattern.tif"
r = io.imread(path)
r = normalize(r)
t = normalize(r)

x = np.arange(r.shape[1]) - 512
y = np.arange(r.shape[0]) - 512
X, Y = np.meshgrid(x, y)
xi = np.array([Y.flatten(), X.flatten()]).astype(np.float32)
r_zmsv = np.sqrt(((r - r.mean())**2).sum()).astype(np.float32)
NablaR_dot_Jac = np.random.rand(8, r.shape[0]*r.shape[1]).astype(np.float32)
H = 2 / r_zmsv**2 * NablaR_dot_Jac.dot(NablaR_dot_Jac.T)
p = np.zeros(8).astype(np.float32)
r = r.flatten().astype(np.float32)
t = t.flatten().astype(np.float32)

xi_gpu = torch.tensor(xi).to(device)
r_gpu = torch.tensor(r).to(device)
t_gpu = torch.tensor(t).to(device)
NablaR_dot_Jac_gpu = torch.tensor(NablaR_dot_Jac).to(device)
H_gpu = torch.tensor(H).to(device)
p_gpu = torch.tensor(p).to(device)

iterations = 1000

print("CPU")
tcpu = timeit.timeit(lambda: main_cpu(r, t, p, NablaR_dot_Jac, H, r_zmsv, xi), number=iterations)
print(f"Time: {tcpu/iterations} seconds/iteration")

print("GPU")
tcpu = timeit.timeit(lambda: main_gpu(r_gpu, t_gpu, p_gpu, NablaR_dot_Jac_gpu, H_gpu, r_zmsv, xi_gpu), number=iterations)
print(f"Time: {tcpu/iterations} seconds/iteration")
