# Script to create an sub-region ANG file and a .txt grain file from the full ang and grain files
import os
import numpy as np
import ebsd_pattern
from skimage import io
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

### USER INPUTS ###
folder = "E:/"
fname = "DED_CoNi90"
# folder = "/Users/jameslamb/Downloads/"
# fname = "DED_CoNi90"
save_extension = "_roi"
roi_start = (0, 0)
roi_dims = (50, 50)
number_of_grain_columns = 11
### END USER INPUTS ###

ang_path = folder + fname + ".ang"
grain_path = folder + fname + ".txt"
up2_path = folder + fname + ".up2"
up2_save_folder = folder + "patterns/"
ang_save_path = folder + fname + save_extension + ".ang"
grain_save_path = folder + fname + save_extension + ".txt"

# Need to grab the header from the ang file and grab the scan dimensions/resolution
print("Parsing ANG file...")
with open(ang_path, 'r') as f:
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

header = open(ang_path).readlines()[:header_end]
data = np.genfromtxt(ang_path, skip_header=header_end)
data = data.reshape(n_rows, n_cols, data.shape[1])

print("  Number of header lines: ", header_end)
print("  Scan dimensions: ", n_rows, n_cols)
print("  Scan resolution: ", y_step, x_step)
print("  Data shape: ", data.shape)
print("  x range: ", data[0, 0, 3], data[-1, -1, 3])
print("  y range: ", data[0, 0, 4], data[-1, -1, 4])

# Choose the location of the ROI
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(data[:, :, 5])
coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')

    global coords
    coords.append((ix, iy))

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

roi_start = (int(coords[-1][1]), int(coords[-1][0]))
print("  ROI start: ", roi_start)

roi_data = data[roi_start[0]:roi_start[0]+roi_dims[0], roi_start[1]:roi_start[1]+roi_dims[1], :]
roi_y, roi_x = np.indices(roi_dims, dtype=float)
roi_x *= x_step
roi_y *= y_step
roi_data[:, :, 3] = roi_x
roi_data[:, :, 4] = roi_y

print("  ROI data shape: ", roi_data.shape)
print("  ROI x range: ", roi_data[0, 0, 3], roi_data[-1, -1, 3])
print("  ROI y range: ", roi_data[0, 0, 4], roi_data[-1, -1, 4])

print("  Writing new ANG file...")
roi_data = roi_data.reshape(-1, roi_data.shape[2])
with open(ang_save_path, 'w') as f:
    for l in header:
        if l[:8] == "# NROWS:":
            l = "# NROWS: {}\n".format(roi_dims[0])
        elif l[:13] == "# NCOLS_EVEN:":
            l = "# NCOLS_EVEN: {}\n".format(roi_dims[1])
        elif l[:12] == "# NCOLS_ODD:":
            l = "# NCOLS_ODD: {}\n".format(roi_dims[1])
        f.write(l)
    for l in roi_data:
        line = "  {:.5f}   {:.5f}   {:.5f}      {:.5f}      {:.5f}    {:.1f}  {:.3f}  {:.0f}      {:.0f} {:.3f}\n".format(*l)
        f.write(line)

print("Parsing grain file...")
with open(grain_path, 'r') as f:
    for i, l in enumerate(f.readlines()):
        if l[0] != "#":
            header_end = i
            break

header = open(grain_path).readlines()[:header_end]
data = np.genfromtxt(grain_path, skip_header=header_end, dtype=str)
phase = data[:, number_of_grain_columns-1:]
if phase.shape[1] > 1:
    phase = np.array(["_".join(p) for p in phase])
data = data[:, :number_of_grain_columns-1].astype(float)
phase = phase.reshape(n_rows, n_cols)
data = data.reshape(n_rows, n_cols, number_of_grain_columns-1)

print("  Number of header lines: ", header_end)
print("  Data shape: ", data.shape)
print("  Phase shape: ", phase.shape)
print("  x range: ", data[0, 0, 3], data[-1, -1, 3])
print("  y range: ", data[0, 0, 4], data[-1, -1, 4])

roi_data = data[roi_start[0]:roi_start[0]+roi_dims[0], roi_start[1]:roi_start[1]+roi_dims[1], :]
roi_phase = phase[roi_start[0]:roi_start[0]+roi_dims[0], roi_start[1]:roi_start[1]+roi_dims[1]]
roi_data[:, :, 3] = roi_x
roi_data[:, :, 4] = roi_y
grain_ids = np.unique(roi_data[:, :, 8])
for i in range(grain_ids.shape[0]):
    if grain_ids[i] == 0:
        continue
    else:
        roi_data[roi_data[:, :, 8] == grain_ids[i], 8] = i

print("  ROI data shape: ", roi_data.shape)
print("  ROI phase shape: ", roi_phase.shape)
print("  ROI x range: ", roi_data[0, 0, 3], roi_data[-1, -1, 3])
print("  ROI y range: ", roi_data[0, 0, 4], roi_data[-1, -1, 4])

print("  Writing new grain file...")
roi_data = roi_data.reshape(-1, roi_data.shape[2])
roi_phase = roi_phase.reshape(-1)
with open(grain_save_path, 'w') as f:
    for l in header:
        f.write(l)
    for l, p in zip(roi_data, roi_phase):
        line = "  {:.5f}   {:.5f}   {:.5f}   {:.5f}   {:.5f}   {:.1f}  {:.3f}  {:.0f}   {:.0f}  {:.3f}  {}\n".format(*l, p)
        f.write(line)

print("Orientation and Grain files created. Exporting patterns...")
ebsd_obj = ebsd_pattern.get_pattern_file_obj(up2_path)
ebsd_obj.read_header()
print("  Number of patterns: ", ebsd_obj.nPatterns)
print("  Pattern dimensions: ", ebsd_obj.patternH, ebsd_obj.patternW)

ii = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
ii = ii[roi_start[0]:roi_start[0]+roi_dims[0], roi_start[1]:roi_start[1]+roi_dims[1]].flatten()
print("  Number of patterns in ROI: ", ii.size)

os.makedirs(up2_save_folder, exist_ok=True)
roi_x = (roi_x.flatten() * 1000).astype(int)
roi_y = (roi_y.flatten() * 1000).astype(int)
for i, index in tqdm(enumerate(ii), desc="Exporting patterns", unit="pats", total=ii.size):
    pat = np.squeeze(ebsd_obj.pat_reader(index, 1))
    pat = np.around(2**16 * (pat - pat.min()) / (pat.max() - pat.min())).astype(np.uint16)
    name = f"{up2_save_folder}{fname}{save_extension}_x{roi_x[i]}y{roi_y[i]}.tiff"
    io.imsave(name, pat)
