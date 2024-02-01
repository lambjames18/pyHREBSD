# Script to create an sub-region ANG file and a .txt grain file from the full ang and grain files
import numpy as np

folder = "E:/James/"
fname = "DED_CoNi90"

# Need to grab the header from the ang file and grab the scan dimensions/resolution


header = open(path).readlines()[:header_end]
data = np.genfromtxt(path)

new_data = np.zeros((scan_size[0], scan_size[1], data.shape[1]), dtype=float).reshape(-1, data.shape[1])
row, col = np.indices(scan_size)
row = row.flatten() * y_step
col = col.flatten() * x_step
new_data[:, 3] = col
new_data[:, 4] = row

with open(save_path, 'w') as f:
    for l in header:
        f.write(l)
    for l in new_data:
        line = "  {:.5f}   {:.5f}   {:.5f}      {:.5f}      {:.5f}    {:.1f}  {:.3f}  {:.0f}      {:.0f} {:.3f}\n".format(*l)
        f.write(line)