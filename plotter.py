import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_singular_values(s, filename, title):
    """Helper function for plotting singular values"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax.plot(np.arange(len(s)) + 1, s)
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_xlim(0, len(s) + 1)
    ax.set_ylabel("Singular Value")
    # ax.set_yscale('log')
    # ax.set_ylim()
    plt.savefig("./figures/" + filename, bbox_inches="tight", pad_inches=0.2)


def plot_explained_variance_ratio(evr, filename, title):
    """Helper function for plotting explained variance ratio"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax.plot(np.arange(len(evr)) + 1, evr)
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_xlim(0, len(evr) + 1)
    ax.set_ylabel("Explained Variance Ratio")
    # ax.set_yscale('log')
    # ax.set_ylim()
    plt.savefig("./figures/" + filename, bbox_inches="tight", pad_inches=0.1)


def plot_reconstructions(
    orig_faces,
    corrupted_faces,
    recon_faces,
    face_index,
    n_pcs,
    pc_index,
    title,
    filename,
    outlier_matrix=None,
):
    """Helper function to plot and compare original and reconstructed images"""
    if outlier_matrix is None:
        fig, axs = plt.subplots(
            nrows=len(face_index),
            ncols=len(pc_index) + 2,
            figsize=(len(face_index), len(pc_index) + 1),
            subplot_kw={"xticks": [], "yticks": []},
        )
    else:
        fig, axs = plt.subplots(
            nrows=len(face_index),
            ncols=len(pc_index) + 3,
            figsize=(len(face_index), len(pc_index) + 1),
            subplot_kw={"xticks": [], "yticks": []},
        )

    for i, fidx in enumerate(face_index):
        axs[i, 0].imshow(orig_faces[fidx], cmap="gray")
        axs[i, 1].imshow(corrupted_faces[fidx], cmap="gray")
        for j, pidx in enumerate(pc_index):
            axs[i, j + 2].imshow(recon_faces[pidx][fidx], cmap="gray")
            # axs[i, j + 1].set_title("Face: {} (n_pcs={})".format(fidx, n_pcs[pidx]))
        if outlier_matrix is not None:
            axs[i, len(pc_index) + 2].imshow(outlier_matrix[pidx][fidx], cmap="gray")

    plt.suptitle(title)
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.show()

    # axs[0, 0].set_title('Original')


def plot_metrics(mse_avg, psnr_avg, ssim_avg, n_pc, filename, figsize=(11, 3)):
    """Helper function to plot metrics for different methods"""
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    # for mse, psnr, ssim in zip(mse_avg, psnr_avg, ssim_avg):
    axs[0].plot(n_pc, mse_avg)
    axs[1].plot(n_pc, psnr_avg)
    axs[2].plot(n_pc, ssim_avg)

    axs[0].set_xlabel("Number of PCs")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True)
    axs[1].set_xlabel("Number of PCs")
    axs[1].set_ylabel("PSNR")
    axs[1].grid(True)
    axs[2].set_xlabel("Number of PCs")
    axs[2].set_ylabel("SSIM")
    axs[2].grid(True)

    plt.legend("PCA", "SSR-R2D-PCA")
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.show()
