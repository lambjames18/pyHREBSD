# Script to create an sub-region ANG file and a .txt grain file from the full ang and grain files
import os
import numpy as np
import ebsd_pattern
from skimage import io
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

### USER INPUTS ###
path = "E:/DED_CoNi90.ang"
save_path = "E:/DED_CoNi90_0.ang"
sharpness_path = "E:/sharpness.npy"

# Need to grab the header from the ang file and grab the scan dimensions/resolution
print("Parsing ANG file...")
with open(path, 'r') as f:
    for i, l in enumerate(f.readlines()):
        if l[0] != "#":
            header_end = i
            break
        elif l[:13] == "# NCOLS_EVEN:":
            n_cols = int(float(l.split(":")[1].replace("\n", "")))
        elif l[:8] == "# NROWS:":
            n_rows = int(float(l.split(":")[1].replace("\n", "")))
        elif l[:8] == "# XSTEP:":
            x_step = float(l.split(":")[1].replace("\n", ""))
        elif l[:8] == "# YSTEP:":
            y_step = float(l.split(":")[1].replace("\n", ""))

header = open(path).readlines()[:header_end]
data = np.genfromtxt(path, skip_header=header_end)
data = data.reshape(n_rows, n_cols, data.shape[1])
s = np.load(sharpness_path)
s = np.around(2**16 * (s - s.min()) / (s.max() - s.min()))
data[:, :, 5] = s


print("  Writing new ANG file...")
data = data.reshape(-1, data.shape[2])
fmts = ["%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.1f", "%.3f", "%.0f", "%.0f", "%.3f", "%.6f", "%.6f", "%.6f"]
space = [3, 5, 5, 7, 7, 7, 3, 3, 7, 4, 7, 7, 7]
with open(save_path, 'w') as f:
    for l in header:
        f.write(l)
    for d in data:
        spaces = [" "*(space[i] - len(str(int(d[i])))) for i in range(len(d))]
        values = [fmts[i] % (d[i]+0.0) for i in range(len(d))]
        line = "".join([spaces[i] + values[i] for i in range(len(d))])
        line = line + "\n"
        # line = "  {:.5f}   {:.5f}   {:.5f}      {:.5f}      {:.5f}    {:.1f}  {:.3f}  {:.0f}      {:.0f} {:.3f}\n".format(*l)
        f.write(line)
