"""Build the README figures: bilateral filter before/after, and CPU vs GPU time.

Run from the repository root:  python scripts/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
for ax, path, title in [
    (axes[0], "TP1/Images/lena_gray.bmp", "Input (512 x 512, grayscale)"),
    (axes[1], "TP1/Images/lena_filtered_cuda.bmp", "Bilateral filter, CUDA"),
]:
    ax.imshow(Image.open(path), cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.tight_layout()
fig.savefig("images/bilateral_filter.png", dpi=150)
plt.close(fig)

# timings reported in the README (512 x 512 image)
fig, ax = plt.subplots(figsize=(5, 2.6))
labels = ["GPU, CUDA (16 x 16 blocks)", "CPU, sequential C"]
times = [3.3, 95.4]
ax.barh(labels, times, color=["#27ae60", "#7f8c8d"])
for y, t in enumerate(times):
    ax.text(t + 1, y, f"{t} ms", va="center")
ax.set_xlim(0, 115)
ax.set_xlabel("Execution time (ms)")
ax.set_title("Bilateral filter: about 29x speedup")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("images/cpu_vs_gpu.png", dpi=150)
