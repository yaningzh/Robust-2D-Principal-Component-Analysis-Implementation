import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calc_metrics(orig_image, recon_image):
    """Calculate and return MSE, PSNR, and SSIM for original and reconstructed images

    Paramters
    ---------
    orig_image : array, shape (n_images, m, n)

    recon_image : array, shape (n_images, m, n)

    Returns
    -------
    mse : array, shape (n_images,)

    psnr : array, shape (n_images,)

    ssim : array, shape (n_images,)
    """
    if orig_image.shape != recon_image.shape:
        return ValueError("ORIG_IMAGE and RECON_IMAGE must have same shape.")
    n_images = orig_image.shape[0]

    # Initialize arrays for metrics
    mse = np.zeros((n_images,))
    psnr = np.zeros((n_images,))
    ssim = np.zeros((n_images,))

    # Evaluate metric for each original and reconstructed image pair
    for i, (orig, recon) in enumerate(zip(orig_image, recon_image)):
        mse[i] = mean_squared_error(orig, recon)
        psnr[i] = peak_signal_noise_ratio(
            orig, recon, data_range=np.max(recon) - np.min(recon)
        )
        ssim[i] = structural_similarity(orig, recon)

    return mse, psnr, ssim
