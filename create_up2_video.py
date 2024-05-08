import numpy as np
import matplotlib.pyplot as plt

import utilities


up2 = "E:/SiGe/ScanA.up2"
pats = utilities.get_patterns(utilities.read_up2(up2))
pats = pats.reshape(-1, pats.shape[-2], pats.shape[-1])[:, ::4, ::4]

for i in range(pats.shape[0]):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.imshow(pats[i], cmap="gray")
    ax.axis("off")
    ax.text(0.01, 0.99, f"Pattern {i}", transform=ax.transAxes, color="white", fontsize=12, ha="left", va="top")
    plt.savefig(f"gif/pattern_{i}.png", dpi=300)
    plt.close(fig)

# Create a video
utilities.make_video("gif/", "ScanA_patterns.mp4", fps=5)