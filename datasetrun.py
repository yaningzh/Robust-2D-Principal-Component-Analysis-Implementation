import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)
import pdb
import sys

# replaces image with random black square as percentage of image
def draw_random_black_occlusion(img, percentage):
    if percentage < 0 or percentage > 1:
        sys.exit("invalid perecentage value")

    img_height, img_width = img.shape
    rect_height = int(img_height * percentage)
    rect_width = int(img_width * percentage)
    max_width_start = img_width - rect_width
    max_height_start = img_height - rect_height
    start_point_width = np.random.randint(0, max_width_start)
    start_point_height = np.random.randint(0, max_height_start)
    out_img = np.copy(img)
    out_img[
        start_point_height : start_point_height + rect_height,
        start_point_width : start_point_width + rect_width,
    ] = 0

    # uncomment to see plots of occluded image
    # plt.imshow(out_img,cmap='gray')
    # plt.show()

    return out_img


# faces = datasets.fetch_olivetti_faces()
# noise_percentage = 0.2
# mse_losses = []
# psnr_losses = []
# ssim_losses = []
# for img in faces.images:
#     noisy_img = draw_random_black_occlusion(img, noise_percentage)
#     # reco_img = reconstruct_img(noisy_img)
#     psnr_losses.append(peak_signal_noise_ratio(img, reco_img))
#     ssim_losses.append(structural_similarity(img, reco_img))
#     mse_losses.append(mean_squared_error(img, reco_img))
# print(np.mean(psnr_losses))
# print(np.mean(ssim_losses))
# print(np.mean(mse_losses))
