# Script to create an sub-region ANG file and a .txt grain file from the full ang and grain files
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import utilities
import Data


class ANG:
    def __init__(self, ang_path: str) -> None:
        self.path = ang_path
        self.read()
        self.format_lut = dict(
            phi1="{:.5f}",
            phi="{:.5f}",
            phi2="{:.5f}",
            x="{:.5f}",
            y="{:.5f}",
            iq="{:.5f}",
            ci="{:.3f}",
            phase="{:.0f}",
            sem="{:.0f}",
            fit="{:.3f}",
        )

    def read(self) -> None:
        with open(self.path, "r") as f:
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
                    self.column_headers = (
                        l.split(":")[1].replace("\n", "").strip().split(", ")
                    )
                    self.column_headers_lower = [
                        h.split(" ")[0].lower() for h in self.column_headers
                    ]
        self.header = open(self.path).readlines()[: self.header_line_count]
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
        leading_space = [
            int(self.data[:, :, i].max() // 10 + 3) for i in range(self.data.shape[2])
        ]
        with open(path, "w") as f:
            for l in self.header:
                f.write(l)
            for d in self.data.reshape(-1, self.data.shape[2]):
                line = (
                    "".join(
                        [
                            " " * (leading_space[i] - int(d[i] // 10))
                            + self.format_lut[self.column_headers_lower[i]].format(d[i])
                            for i in range(len(d))
                        ]
                    )
                    + "\n"
                )
                f.write(line)


def main(idx: int, pat_obj: Data.UP2) -> float:
    """Main function to process a single pattern.

    Args:
        idx (int): Index of the pattern to process.
        pat_obj (Data.UP2): Pattern object.

    Returns:
        sharpness: Sharpness of the pattern.
    """
    pats = pat_obj.read_pattern(idx)
    f = np.fft.fft2(pats)
    f = np.real(f)
    fshift = np.fft.fftshift(f)
    AF = abs(fshift)
    thresh = AF.max() / 2500
    th = (fshift > thresh).sum()
    return th / (pats.shape[0] * pats.shape[1])


ang_paths = [
    "E:/cells/Ortho_20240229_24197/CoNi90-OrthoCells_20240229_24197_scan3.ang",
    "E:/cells/CoNi90-OrthoCells_20240320_27061_scan3.ang",
    "E:/cells/CoNi90-ParallelCells_20240320_27064_scan6.ang",
    "E:/cells/Parallel_20240229_24207/CoNi90-ParallelCells_20240229_24207_scan6.ang",
    "E:/cells/CoNi90_full.ang",
][-1]

up2_paths = [
    "E:/cells/Ortho_20240229_24197/CoNi90-OrthoCells_20240229_24197_scan3_256x256.up2",
    "E:/cells/CoNi90-OrthoCells_20240320_27061_scan3_1024x1024.up2",
    "E:/cells/CoNi90-ParallelCells_20240320_27064_scan6_1024x1024.up2",
    "E:/cells/Parallel_20240229_24207/CoNi90-ParallelCells_20240229_24207_scan6_256x256.up2",
    "F:/CoNi90/DED_CoNi90.up2",
][-1]

ang_path = ang_paths
up2_path = up2_paths

up2 = Data.UP2(up2_path)
ang = ANG(ang_path)

print("UP2 path:", up2_path)
print("ANG path:", ang_path)
print("Pattern shape:", up2.patshape)
print("Number of patterns:", up2.nPatterns)
print("Scan shape:", ang.shape)

# with utilities.tqdm_joblib(
#     tqdm(total=up2.nPatterns, desc="Patterns processed")
# ) as progress_bar:
#     sharpness = Parallel(n_jobs=10)(
#         delayed(main)(idx, up2) for idx in range(up2.nPatterns)
#     )

# sharpness = np.array(sharpness)
# np.save("temp_sharpness.npy", sharpness)
sharpness = np.load("F:/CoNi90/sharpness.npy")
sharpness = sharpness.reshape(ang.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(sharpness)
ax[0].set_title("Sharpness")
ax[1].imshow(ang.data[:, :, 6])
ax[1].set_title("CI")
plt.tight_layout()
plt.show()

ang.data[:, :, 5] = sharpness
folder = os.path.dirname(ang_path)
filename = os.path.basename(ang_path).split(".")[0]
out_path = os.path.join(folder, filename + "_sharpness.ang")
ang.write(out_path)
print("Sharpness ang file written to:", out_path)
# os.remove("temp_sharpness.npy")
