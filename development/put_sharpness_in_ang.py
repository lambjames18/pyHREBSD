# Script to create an sub-region ANG file and a .txt grain file from the full ang and grain files
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import utilities

class ang_file:
    def __init__(self, ang_path: str) -> None:
        self.path = ang_path
        self.read()
        self.format_lut = dict(phi1="{:.5f}", phi="{:.5f}", phi2="{:.5f}", x="{:.5f}", y="{:.5f}", iq="{:.5f}", ci="{:.3f}", phase="{:.0f}", sem="{:.0f}", fit="{:.3f}")

    def read(self) -> None:
        with open(self.path, 'r') as f:
            for i, l in enumerate(f.readlines()):
                if l[0] != "#":
                    self.header_line_count = i
                    break
                elif l[:13] == "# NCOLS_EVEN:":
                    self.n_cols = int(float(l.split(":")[1].replace("\n", "")))
                elif l[:8] == "# NROWS:":
                    self.n_rows = int(float(l.split(":")[1].replace("\n", "")))
                elif l[:8] == "# XSTEP:":
                    self.x_step = float(l.split(":")[1].replace("\n", ""))
                elif l[:8] == "# YSTEP:":
                    self.y_step = float(l.split(":")[1].replace("\n", ""))
                elif l[:17] == "# COLUMN_HEADERS:":
                    self.column_headers = l.split(":")[1].replace("\n", "").strip().split(", ")
                    self.column_headers_lower = [h.split(" ")[0].lower() for h in self.column_headers]
        self.header = open(self.path).readlines()[:self.header_line_count]
        self.data = np.genfromtxt(self.path, skip_header=self.header_line_count)
        self.data = self.data.reshape(self.n_rows, self.n_cols, self.data.shape[1])
        self.shape = (self.n_rows, self.n_cols)
        # print("line 24:", self.header[24])

    def crop(self, x0, x1, y0, y1) -> None:
        """Crops the data to the specified bounds.
        This will also update the header and data attributes.
        This cannot be undone."""
        self.data = self.data[y0:y1, x0:x1]
        # Resample the x/y positions in data
        x_index = self.column_headers_lower.index("x")
        y_index = self.column_headers_lower.index("y")
        y, x = np.indices(self.data.shape[:2], dtype=float)
        y *= self.y_step
        x *= self.x_step
        self.data[:, :, x_index] = x
        self.data[:, :, y_index] = y
        # Update the header info
        self.n_rows, self.n_cols = self.data.shape[:2]
        for i in range(len(self.header)):
            if self.header[i][:8] == "# NROWS:":
                self.header[i] = "# NROWS: {}\n".format(self.n_rows)
            elif self.header[i][:13] == "# NCOLS_EVEN:":
                self.header[i] = "# NCOLS_EVEN: {}\n".format(self.n_cols)
            elif self.header[i][:12] == "# NCOLS_ODD:":
                self.header[i] = "# NCOLS_ODD: {}\n".format(self.n_cols)

    def write(self, path: str):
        leading_space = [int(self.data[:, :, i].max() // 10 + 3) for i in range(self.data.shape[2])]
        with open(path, 'w') as f:
            for l in self.header:
                f.write(l)
            for d in self.data.reshape(-1, self.data.shape[2]):
                line = "".join([" " * (leading_space[i] - int(d[i] // 10)) + self.format_lut[self.column_headers_lower[i]].format(d[i]) for i in range(len(d))]) + "\n"
                f.write(line)


class up2_file:
    
    def __init__(self, up2_path: str) -> None:
        self.path = up2_path
        self.scan_shape = None
        self.read()

    def read(self) -> None:
        with open(self.path, "rb") as upFile:
            chunk_size = 4
            tmp = upFile.read(chunk_size)
            self.FirstEntry = struct.unpack('i', tmp)[0]
            tmp = upFile.read(chunk_size)
            sz1 = struct.unpack('i', tmp)[0]
            tmp = upFile.read(chunk_size)
            sz2 = struct.unpack('i', tmp)[0]
            tmp = upFile.read(chunk_size)
            self.bitsPerPixel = struct.unpack('i', tmp)[0]
            sizeBytes = os.path.getsize(self.path) - 16
            self.sizeString = str(round(sizeBytes / 1e6, 1)) + " MB"
            bytesPerPixel = 2
            # print("Header:", self.FirstEntry, sz1, sz2, self.bitsPerPixel)
            self.nPatterns = int((sizeBytes/bytesPerPixel) / (sz1 * sz2))
            self.patshape = (sz1, sz2)
        self.pidx = np.arange(self.nPatterns)


    def get_patterns(self, idx: None) -> tuple:
        """Read in patterns from a pattern file object.

        Args:
            idx (np.ndarray | list | tuple): Indices of patterns to read in. If None, reads in all patterns.

        Returns:
            np.ndarray: Patterns."""
        # Handle inputs
        if idx is None:
            idx = range(self.nPatterns)
            reshape = False
        else:
            idx = np.asarray(idx)
            reshape = False
            if idx.ndim >= 2:
                reshape = True
                out_shape = idx.shape + self.patshape
                idx = idx.flatten()

        # Read in the patterns
        start_byte = np.int64(16)
        pattern_bytes = np.int64(self.patshape[0] * self.patshape[1] * 2)
        pats = np.zeros((len(idx), *self.patshape), dtype=np.uint16)
        with open(self.path, "rb") as datafile:
            for i in tqdm(range(len(idx)), desc="Reading patterns", unit="pats"):
            # for i in range(len(idx)):
                pat = np.int64(idx[i])
                seek_pos = np.int64(start_byte + pat * pattern_bytes)
                datafile.seek(seek_pos)
                pats[i] = np.frombuffer(
                    datafile.read(self.patshape[0] * self.patshape[1] * 2),
                    dtype=np.uint16,
                ).reshape(self.patshape)

        # Reshape the patterns
        pats = np.squeeze(pats)
        if reshape:
            return pats.reshape(out_shape)
        else:
            return pats

    def set_scan_shape(self, shape: tuple) -> None:
        if shape[0] * shape[1] != self.nPatterns:
            raise ValueError("The number of patterns in the scan shape must match the number of patterns in the file.")
        self.scan_shape = shape
        self.pidx = self.pidx.reshape(shape)

    def crop(self, x0, x1, y0: None, y1:None) -> None:
        if self.scan_shape is not None and (y0 is None or y1 is None):
            raise ValueError("If the scan shape is set, y0 and y1 must be specified.")
        elif self.scan_shape is None and (y0 is not None or y1 is not None):
            raise ValueError("If y0 and y1 are specified, the scan shape must be set.")
        if self.scan_shape is not None:
            self.pidx = self.pidx[y0:y1, x0:x1]
            self.nPatterns = self.pidx.size
        else:
            self.pidx = self.pidx[x0:x1]
            self.nPatterns = self.pidx.size

    def write(self, path: str):

        # Get the pattern dimensions
        sz1, sz2 = self.patshape

        # Set reading parameters
        start_byte = np.int64(16)
        pattern_bytes = np.int64(sz1 * sz2 * 2)
        pidx = self.pidx.flatten()

        # Open the file
        with open(path, "wb") as writeFile:
            # Write the header
            writeFile.write(struct.pack('i', 1))
            writeFile.write(struct.pack('i', sz1))
            writeFile.write(struct.pack('i', sz2))
            writeFile.write(struct.pack('i', 16))
            with open(self.path, "rb") as readFile:
                for i in pidx:
                    seek_pos = start_byte + np.int64(i) * pattern_bytes
                    readFile.seek(seek_pos)
                    # pat = np.frombuffer(readFile.read(pattern_bytes), dtype=np.uint16)
                    # writeFile.write(pat.tobytes())
                    writeFile.write(readFile.read(pattern_bytes))


### USER INPUTS ###
up2 = "E:/cells/CoNi90-ParallelCells_20240320_27064_scan6_1024x1024.up2"
ang = "E:/cells/CoNi90-ParallelCells_20240320_27064_scan6.ang"

ang_obj = ang_file(ang)
iq_index = ang_obj.column_headers_lower.index("iq")
ci_index = ang_obj.column_headers_lower.index("ci")
print(ang_obj.data[:, :, iq_index].min(), ang_obj.data[:, :, iq_index].max())

up2_obj = up2_file(up2)
pidxes = np.array_split(np.arange(up2_obj.nPatterns), 10)
sharpness = np.zeros(up2_obj.nPatterns)
for pidx in pidxes:
    pats = up2_obj.get_patterns(pidx)
    s = utilities.get_sharpness(pats)
    sharpness[pidx] = s

sharpness = sharpness.reshape(ang_obj.shape)

np.save("E:/cells/CoNi90-ParallelCells_20240320_27064_scan6_sharpness.npy", sharpness)
# sharpness = np.load("E:/cells/CoNi90-ParallelCells_20240320_27064_scan6_sharpness.npy")

ang_obj.data[:, :, iq_index] = sharpness

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(ang_obj.data[:, :, iq_index])
ax[0].set_title("IQ (Sharpness)")
ax[1].imshow(ang_obj.data[:, :, ci_index])
ax[1].set_title("CI")
plt.show()

print(ang_obj.data[:, :, iq_index].min(), ang_obj.data[:, :, iq_index].max())

ang_obj.write("E:/cells/CoNi90-ParallelCells_20240320_27064_scan6_sharpness.ang")

# ang_obj = ang_file(ang)
# ang_obj.crop(0, 50, 150, 200)
# ang_obj.write("E:/GaN/GaN_0.ang")

# up2_obj = up2_file(up2)
# up2_obj.set_scan_shape((200, 200))
# up2_obj.crop(0, 50, 150, 200)
# up2_obj.write("E:/GaN/GaN_0.up2")


