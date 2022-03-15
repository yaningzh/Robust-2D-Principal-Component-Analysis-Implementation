"""Preprocessing to generate corrupted images"""
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import skimage
from PIL import Image


def occlude(im, block_size):
    m, n = im.shape
    return None


# Load data
image_dir = "./datasets/yalefaces/subject*.*"
files = glob.glob(image_dir)
images = np.asarray([np.asarray(Image.open(f)) for f in files])
n_images, m, n = np.shape(images)

# Plot 10 random images
nrows = 2
ncols = 5
fig, axs = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(8, 3), gridspec_kw={"wspace": 0, "hspace": 0}
)
axs = axs.reshape(-1)
for i, idx in enumerate(random.sample(range(n_images), nrows * ncols)):
    axs[i].imshow(images[idx], cmap="gray")
    axs[i].set_xticks([])
    axs[i].set_yticks([])

fig.tight_layout()
plt.savefig("./figures/yalefaces.png", bbox_inces="tight", dpi=200)
plt.show()

# Paramters for generating corrupted images (s&p)
mean = 0
var = 0.01
amount = 0.05

# Paramters for generating corrupted images (occlusion)
block_size = [10, 10]

# Generate corrupted images
for i, im in enumerate(images):
    # Salt and pepper
    image_snp = skimage.util.random_noise(im, mode="s&p", amount=amount)

    # Save to cache
    np.save("./cache/yalefaces_{}_snp_{}.npy".format(i, amount), image_snp)

    # Plot and save image with S&P
    plt.imshow(image_snp, cmap="gray")
    plt.savefig(
        "./figures/snp/yalefaces_{}_snp_{}.png".format(i, amount),
        bbox_inces="tight",
        dpi=200,
    )

    # Occlusion
    # image_blocked = occlude(im, block_size)
    # plt.savefig(
    #     "./figures/occlude/yalefaces_occlude_{}.png".format(i),
    #     bbox_inces="tight",
    #     dpi=200,
    # )

# TODO : Function to generate images with uniform occlusions (pixels black or white)
# TODO : Function to generate images with white noise
# TODO : Generate occluded images to be used in experiments
