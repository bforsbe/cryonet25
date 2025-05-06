# Largely written by chatGPT with essential modifications
# https://chatgpt.com/share/6819a802-a900-8003-a179-fb27806c5be8
# can easily be run on e.g. https://python-fiddle.com/ or similar,
# as here https://python-fiddle.com/saved/20ccce2b-32e2-4feb-9670-5f756829c5b4

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage import data, color
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Parameters
N = 500
shift_std = 2.0
image_size = 256

# Create circular low-pass mask
def circular_lowpass_mask(shape):
    yy, xx = np.indices(shape)
    center = np.array(shape) // 2
    radius = shape[0] // 2  # maximal inscribed circle
    dist = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    return dist <= radius

def apply_circular_lowpass(img):
    ft = fft2(img)
    ft_shifted = fftshift(ft)
    mask = circular_lowpass_mask(img.shape)
    ft_filtered = ft_shifted * mask
    return np.real(ifft2(ifftshift(ft_filtered)))

# Load and preprocess base image
base = color.rgb2gray(data.astronaut())
base = resize(base, (image_size, image_size), anti_aliasing=True)
base = apply_circular_lowpass(base)  # apply low-pass to base image

# Generate translated images with 'wrap' and apply low-pass
translated_images = []
for _ in range(N):
    dx, dy = np.random.normal(0, shift_std, 2)
    translated = shift(base, shift=(dy, dx), mode='wrap')  # use wrap
    translated = apply_circular_lowpass(translated)
    translated_images.append(translated)

avg_image = np.mean(translated_images, axis=0)

# Power spectrum function
def power_spectrum(img):
    ft = fft2(img)
    return np.abs(fftshift(ft))**2

# Radial averaging function
def radial_average(image):
    y, x = np.indices(image.shape)
    center = np.array(image.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    radial_mean = np.bincount(r.ravel(), image.ravel()) / np.bincount(r.ravel())
    return radial_mean

# Compute power spectra
ps_base = power_spectrum(base)
ps_avg = power_spectrum(avg_image)
r_base = radial_average(ps_base)
r_avg = radial_average(ps_avg)

# Estimate blur transfer function from ratio
eps = 1e-4
blur_est_1d = np.sqrt(r_avg / (r_base + eps))

# Build 2D blur estimate from 1D radial estimate
yy, xx = np.indices((image_size, image_size))
center = image_size // 2
rr = np.sqrt((xx - center)**2 + (yy - center)**2).astype(int)
blur_est_2d = blur_est_1d[rr]

# Deconvolution
deconv_ft = fft2(avg_image) / (fftshift(blur_est_2d) + eps)
deconv_image = np.real(ifft2(deconv_ft))
deconv_image = apply_circular_lowpass(deconv_image)  # optional

# Compute power spectrum after deconvolution
ps_deconv = power_spectrum(deconv_image)
r_deconv = radial_average(ps_deconv)

# Plotting: Images and Spectra
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].imshow(base, cmap='gray')
axs[0, 0].set_title('Original (Low-passed)')
axs[0, 1].imshow(avg_image, cmap='gray')
axs[0, 1].set_title(f'Averaged (N={N})')
axs[0, 2].imshow(deconv_image, cmap='gray')
axs[0, 2].set_title('Deconvolved')

axs[1, 0].imshow(np.log1p(ps_base), cmap='viridis')
axs[1, 0].set_title('Original Power Spectrum')
axs[1, 1].imshow(np.log1p(ps_avg), cmap='viridis')
axs[1, 1].set_title('Averaged Power Spectrum')
axs[1, 2].imshow(np.log1p(ps_deconv), cmap='viridis')
axs[1, 2].set_title('Deconvolved Power Spectrum')

for ax in axs.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()

# Plot: Radial Power Spectra
plt.figure(figsize=(6, 4))
plt.plot(r_base, label='Original', lw=2)
plt.plot(r_avg, label='Averaged', lw=2)
plt.plot(r_deconv, label='Deconvolved', lw=2)
plt.ylim( 1e-1, 1e9)
plt.yscale('log')
plt.xlabel('Spatial Frequency (radial index)')
plt.ylabel('Power Spectrum (log scale)')
plt.title('Radially Averaged Power Spectra')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# Plot: 1D Blur Estimate
plt.figure(figsize=(6, 3.5))
plt.plot(blur_est_1d, color='orange', lw=2)
plt.ylim(0, 1.1)
plt.xlabel('Spatial Frequency (radial index)')
plt.ylabel('Blur Transfer Function')
plt.title('1D Blur Estimate from Averaging')
plt.grid(True)
plt.tight_layout()
plt.show()
