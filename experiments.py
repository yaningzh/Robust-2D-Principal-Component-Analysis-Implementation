"""Experiments for structured sparsity regularized robust 2D principal
component analysis.
"""
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
from SSRR2DPCA import SSRR2DPCA, ssrr2dpca, reconstruct
from plotter import *
from metrics import calc_metrics
from datasetrun import draw_random_black_occlusion

# ##########
# Load data - Yale Face Database
image_dir = "./datasets/yalefaces/subject*.*"
files = glob.glob(image_dir)
faces = np.asarray([np.asarray(Image.open(f)) for f in files]).astype(float)
n_samples, m, n = np.shape(faces)

# #################################################################
# Load data - AT&T Faces
# faces, _ = fetch_olivetti_faces(return_X_y=True)
# n_samples, n_features = faces.shape
# faces -= faces.mean(axis=0)
# # faces_centered = faces - faces.mean(axis=0)
# # faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
# print("AT&T Faces dataset consists of %d faces" % n_samples)
# m, n = 64, 64

# #################################################################
# Generate occluded face images
corrupted_faces = np.copy(faces.reshape((n_samples, m, n)))
corrupted_samples = 0.2
noise_percentage = 0.3
idx2corrupt = np.random.choice(
    range(n_samples), int(corrupted_samples * n_samples), replace=False
)
for ii in idx2corrupt:
    corrupted_faces[ii] = draw_random_black_occlusion(
        corrupted_faces[ii], noise_percentage
    )

# # Plot corrupted image examples
# fig, axs = plt.subplots(
#     nrows=1,
#     ncols=5,
#     figsize=(8, 2),
#     subplot_kw={"xticks": [], "yticks": []},
# )
# for i, idx in enumerate(np.random.choice(idx2corrupt, 5)):
#     axs[i].imshow(corrupted_faces[idx], cmap="gray")

# plt.savefig("./figures/corrupted_images.png", bbox_inches="tight", dpi=200)
# plt.show()

# #################################################################
# Fit model to data using PCA with various n_pcs
n_components = np.arange(5, 80, 5)
mse_pca_avg = []
psnr_pca_avg = []
ssim_pca_avg = []
pca_recon_all = []

for n_pc in n_components:
    pca = PCA(n_components=n_pc)
    X_transformed_pca = pca.fit_transform(corrupted_faces.reshape((n_samples, m * n)))
    pca_pc, pca_evr, pca_sv = (
        pca.components_,
        pca.explained_variance_ratio_,
        pca.singular_values_,
    )

    # Reconstruct images using PCA
    pca_recon = pca.inverse_transform(X_transformed_pca).reshape((n_samples, m, n))

    # Calculate metrics for PCA
    mse_pca, psnr_pca, ssim_pca = calc_metrics(
        faces.reshape((n_samples, m, n)), pca_recon
    )

    # Append average MSE, PSNR, SSIM
    mse_pca_avg.append(np.mean(mse_pca))
    psnr_pca_avg.append(np.mean(psnr_pca))
    ssim_pca_avg.append(np.mean(ssim_pca))

    # Append reconstructions
    pca_recon_all.append(pca_recon)

# Plot reconstructions with various numbers of PCs
face_indices = np.random.choice(idx2corrupt, size=5, replace=False)
pc_indices = np.arange(0, len(n_components), 4)
plot_reconstructions(
    faces.reshape((n_samples, m, n)),
    corrupted_faces,
    pca_recon_all,
    face_indices,
    n_components,
    pc_indices,
    "PCA reconstruction",
    "./figures/PCA_recon.png",
)

# #################################################################
# Fit model to data using SSR-2D-PCA
scale = [10, 15, 20]

mse_sr2pca_avg = []
psnr_sr2pca_avg = []
ssim_sr2pca_avg = []
sr2pca_recon_all = []
sr2pca_E_all = []
npcs_sr2pca = []

for i, s in enumerate(scale):
    ssrU, ssrV, ssrS, ssrE = ssrr2dpca(corrupted_faces, scale=s)
    sr2pca_recon = reconstruct(ssrU, ssrV, ssrS, ssrE)

    # Calculate metrics for SSR-R2D-PCA
    mse_sr2pca, psnr_sr2pca, ssim_sr2pca = calc_metrics(
        faces.reshape((n_samples, m, n)), sr2pca_recon
    )

    # Append stats
    mse_sr2pca_avg.append(np.mean(mse_sr2pca))
    psnr_sr2pca_avg.append(np.mean(psnr_sr2pca))
    ssim_sr2pca_avg.append(np.mean(ssim_sr2pca))
    npcs_sr2pca.append(ssrU.shape[1] + ssrV.shape[1])

    # Append reconstructions
    sr2pca_recon_all.append(sr2pca_recon)
    sr2pca_E_all.append(ssrE)

# Plot reconstructions from SR2PCA
plot_reconstructions(
    faces.reshape((n_samples, m, n)),
    corrupted_faces,
    sr2pca_recon_all,
    face_indices,
    npcs_sr2pca,
    np.array(np.arange(len(scale))),
    "SSR-R2D-PCA reconstruction",
    "./figures/ssrr2dpca_recon_{}.png".format(scale),
    outlier_matrix=sr2pca_E_all,
)

# #################################################################
# Plot metrics
plot_metrics(
    mse_pca_avg,
    psnr_pca_avg,
    ssim_pca_avg,
    n_components,
    "./figures/metrics_pca.png",
)

plot_metrics(
    mse_sr2pca_avg,
    psnr_sr2pca_avg,
    ssim_sr2pca_avg,
    npcs_sr2pca,
    "./figures/metrics_sr2pca.png",
)
