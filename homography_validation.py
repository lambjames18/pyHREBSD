import numpy as np
import matplotlib.pyplot as plt
import warp

size = (480, 640)
images = np.zeros(size, dtype=np.float32)
images[size[0]//4:3*size[0]//4, size[1]//4:3*size[1]//4] = 1.0
images = np.vstack([images.reshape(1, *size) for i in range(9)])
# images[-1] *= 0.0

homographies = np.array([
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 50., 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 50., 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
homographies *= -1

l = [r"$h_{11}$", r"$h_{12}$", r"$h_{13}$", r"$h_{21}$", r"$h_{22}$", r"$h_{23}$", r"$h_{31}$", r"$h_{32}$"]

ratio = size[0] / size[1]
fig, ax = plt.subplots(3, 3, figsize=(8, 8 * ratio))
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
ax = ax.ravel()
for i in range(9):
    im = images[i]
    im_warped = warp.deform_image(im, homographies[i], kx=5, ky=5)
    im_warped[im_warped < 0.5] = 0.0
    im_warped[im_warped > 1.0] = 1.0
    im = np.dstack((im.reshape(size), im_warped.reshape(size), np.zeros((size))))
    ax[i].imshow(im, cmap="Greys_r")
    ax[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    if i != 8:
        ax[i].text(size[1]//2, size[0]//2, f"{l[i]} = {homographies[i, i]}", va="center", ha="center", color="black")
    else:
        ax[i].text(size[1]//2, size[0]//2, f"Image\n{(size[1], size[0])}\n\nSquare\n{(size[1]//2, size[0]//2)}", va="center", ha="center", color="black")

plt.show()