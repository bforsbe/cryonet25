# Largely written by chatGPT with essential modifications
# https://chatgpt.com/share/6819f9a2-6d0c-8003-a2ab-56bc19051e1a
# can easily be run on e.g. https://python-fiddle.com/ or similar,
# as here https://python-fiddle.com/saved/aa55b3c9-9bc1-4987-8877-162d408a011d

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.fft import fft2, fftshift
from skimage.draw import disk

def create_circle_image(image_size, radius):
    image = np.zeros((image_size, image_size), dtype=np.float32)
    center = (image_size // 2, image_size // 2)
    rr, cc = disk(center, radius, shape=image.shape)
    image[rr, cc] = 1.0
    return image

def add_noise(image, noise_variance):
    noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
    return image + noise

def compute_autocorrelation(noisy_image, reference_image):
    return correlate2d(noisy_image, reference_image, mode='same', boundary='wrap')

def compute_spectral_snr_radial(noisy_image, signal_image):
    F_signal = fft2(signal_image)
    F_noisy = fft2(noisy_image)
    signal_power = np.abs(F_signal)**2
    noise_power = np.abs(F_noisy - F_signal)**2
    ssnr_2d = np.divide(signal_power, noise_power, out=np.zeros_like(signal_power), where=noise_power != 0)
    ssnr_2d = fftshift(ssnr_2d)

    center = np.array(ssnr_2d.shape) // 2
    y, x = np.indices(ssnr_2d.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)

    radial_sum = np.bincount(r.ravel(), weights=ssnr_2d.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = np.divide(radial_sum, radial_count, out=np.zeros_like(radial_sum), where=radial_count != 0)
    return radial_profile

def compute_fsc_from_ssnr(ssnr):
    """Estimate FSC from SSNR using the standard relationship."""
    return np.divide(ssnr, ssnr + 1.0, out=np.zeros_like(ssnr), where=(ssnr + 1) != 0)

def plot_results(original, noisy, autocorr, radial_ssnr, fsc):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    for j in range(1, 4):
        axs[0, j].axis('off')

    axs[1, 0].imshow(noisy, cmap='gray')
    axs[1, 0].set_title("Noisy Image")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(autocorr, cmap='hot')
    axs[1, 1].set_title("Autocorrelation")
    axs[1, 1].axis('off')

    nyquist = 0.5
    freqs = np.linspace(0, nyquist, len(radial_ssnr))

    axs[1, 2].plot(freqs, radial_ssnr)
    axs[1, 2].set_yscale("log")
    axs[1, 2].set_title("Radial SSNR (log scale)")
    axs[1, 2].set_xlabel("Fractional Nyquist Frequency")
    axs[1, 2].set_ylabel("SSNR")
    axs[1, 2].grid(True)

    axs[1, 3].plot(freqs, fsc)
    axs[1, 3].set_ylim(0, 1.05)
    axs[1, 3].set_title("Estimated FSC")
    axs[1, 3].set_xlabel("Fractional Nyquist Frequency")
    axs[1, 3].set_ylabel("FSC")
    axs[1, 3].grid(True)

    plt.tight_layout()
    plt.show()

# Parameters
image_size = 128
circle_radius = 30
noise_variance = 0.1

# Processing
original_img = create_circle_image(image_size, circle_radius)
noisy_img = add_noise(original_img, noise_variance)
autocorr_img = compute_autocorrelation(noisy_img, original_img)
radial_ssnr = compute_spectral_snr_radial(noisy_img, original_img)
fsc = compute_fsc_from_ssnr(radial_ssnr)

# Plot
plot_results(original_img, noisy_img, autocorr_img, radial_ssnr, fsc)
