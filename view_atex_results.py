import numpy as np
import matplotlib.pyplot as plt
import utilities


header_lines = 17
path = "E:/SiGe/ScanA____EXPORT_PIXELS.txt"
# path = "E:/SiGe/ScanA____AdaptPC_EXPORT_PIXELS.txt"

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

e_tetragonal = (e11 + e22) / 2 - e33

color = np.array([(254, 188, 17) for i in range(len(ids))])
color[e22 < -0.001] = (0, 54, 96)
color = color / 255

fig, ax = plt.subplots(3, 3, figsize=(15, 14))
ax[0, 0].scatter(x, e11, c=color, marker="s", label=r"$\epsilon_{11}$")
ax[0, 1].scatter(x, e12, c=color, marker="s", label=r"$\epsilon_{12}$")
ax[0, 2].scatter(x, e13, c=color, marker="s", label=r"$\epsilon_{13}$")
ax[1, 0].axis("off")
ax[1, 1].scatter(x, e22, c=color, marker="s", label=r"$\epsilon_{22}$")
ax[1, 2].scatter(x, e23, c=color, marker="s", label=r"$\epsilon_{23}$")
# ax[2, 0].scatter(x, dis, c=color, marker="s", label=r"$\phi$")
ax[2, 0].axis("off")
ax[2, 1].scatter(x, e_tetragonal, c=color, marker="s", label=r"$\epsilon_{tetragonal}$")
ax[2, 2].scatter(x, e33, c=color, marker="s", label=r"$\epsilon_{33}$")


for a in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2], ax[2, 2], ax[2, 1]]:
    a.set_ylim(-0.0018, 0.0018)
    # a.set_ylim(-0.018, 0.018)
    utilities.standardize_axis(a)
    utilities.make_legend(a)

# ax[2, 0].set_ylim(0, 0.05)
# standardize_axis(ax[2, 0])
# make_legend(ax[2, 0],loc="lower right")

plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.07, right=0.99, top=0.99, bottom=0.05)
plt.savefig("E:/SiGe/ScanA_ATEX_results_test.png", dpi=300)
