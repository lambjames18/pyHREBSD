import numpy as np
import matplotlib.pyplot as plt
import utilities

header_lines = 23
geometry = "CS1-0-0-0_CSd-0-90-0"
geometry = "CS1-0-0-0_CSd-0-90-0_NTF"
# geometry = "CS1-180-180-180_CSd-0-90-0"
geometry = "EDAX2_CSd-0-90-0"
geometry = "EDAX2_CSd-180-90-180"
path = "E:/SiGe/a-C03-scan/ScanA_{}_output.txt".format(geometry)

data = np.loadtxt(path, skiprows=header_lines, delimiter="\t", encoding="utf-8")
ids = data[:, 0]
x = data[:, 1]
y = data[:, 2]
dis = data[:, 3]
e11 = data[:, 4]
e12 = data[:, 5]
e13 = data[:, 6]
e22 = data[:, 7]
e23 = data[:, 8]
e33 = data[:, 9]
iterations = data[:, 10]
residuals = data[:, 11]
init_residuals = data[:, 12]
w13 = data[:, 13]
w21 = data[:, 14]
w32 = data[:, 15]

e_tetragonal = (e11 + e22) / 2 - e33

E = np.hstack([e11[:, None], e12[:, None], e13[:, None],
               e12[:, None], e22[:, None], e23[:, None],
               e13[:, None], e23[:, None], e33[:, None]]).reshape(-1, 3, 3)

rm_z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
rm_x180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
rm_x90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
rm_y180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
rm_y90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
rm_y270 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
rm = rm_z180
Eprime = np.matmul(np.matmul(rm, E), rm.T)

e11 = Eprime[:, 0, 0]
e12 = Eprime[:, 0, 1]
e13 = Eprime[:, 0, 2]
e22 = Eprime[:, 1, 1]
e23 = Eprime[:, 1, 2]
e33 = Eprime[:, 2, 2]
e_tetragonal = e33 - (e11 + e22) / 2

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].plot(x, e11, c="r", label=r"$\epsilon_{11}$")
ax[0, 0].plot(x, e22, c="g", label=r"$\epsilon_{22}$")
ax[0, 0].plot(x, e33, c="b", label=r"$\epsilon_{33}$")
ax[0, 1].plot(x, e12, c="m", label=r"$\epsilon_{12}$")
ax[0, 1].plot(x, e13, c="y", label=r"$\epsilon_{13}$")
ax[0, 1].plot(x, e23, c="c", label=r"$\epsilon_{23}$")
ax[0, 2].plot(x, e_tetragonal, c="k", label=r"$\epsilon_{tetragonal}$")
ax[1, 0].plot(x, residuals, c="k", label="Residuals")
ax[1, 1].plot(x, iterations, c="k", label="Num Iterations")
ax[1, 2].plot(x, w13, c="tab:orange", label=r"$\omega_{13}$")
ax[1, 2].plot(x, w21, c="tab:purple", label=r"$\omega_{21}$")
ax[1, 2].plot(x, w32, c="tab:brown", label=r"$\omega_{32}$")

bound = 0.02
for a in [ax[0, 0], ax[0, 1], ax[0, 2]]:
    a.set_ylim(-bound, bound)

for a in ax.flatten():
    utilities.standardize_axis(a)
    utilities.make_legend(a)

plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.08, right=0.99, top=0.95, bottom=0.05)
plt.savefig("E:/SiGe/ScanA_{}_results.png".format(geometry), dpi=300)
