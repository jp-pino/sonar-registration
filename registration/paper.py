import time
import numba
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, resize, warp, SimilarityTransform

from utils import image


def fft(data):
    return np.fft.fftshift(np.fft.fft2(data))


def to_magnitude_and_phase(data):
    return np.abs(data), np.angle(data)


def log_polar_transform(data, radius=None, order=None):
    if radius is None:
        radius = data.shape[0] // 2
    shape = data.shape
    return warp_polar(data, radius=radius, scaling='log', order=order, output_shape=shape)


def register(a, b, mask, force_scale=True):
    # Prepare and apply mask
    time_start = time.time()
    # Add black border to the mask
    mask = np.pad(mask, 100, mode='constant', constant_values=0)
    # Resize mask to image size
    mask = resize(mask, a.shape, anti_aliasing=True)
    # Gaussian blur the mask
    mask = gaussian_filter(mask, sigma=10)
    # Apply mask
    masked_a = blurred_a * mask
    masked_b = blurred_b * mask
    print(f"  > Masking took {time.time() - time_start} seconds")

    # Apply Fourier Transform
    time_start = time.time()
    fft_a, phase_a = to_magnitude_and_phase(fft(masked_a))
    fft_b, phase_b = to_magnitude_and_phase(fft(masked_b))
    print(f"  > Fourier Transform took {time.time() - time_start} seconds")

    # Apply Log Polar Transform
    time_start = time.time()
    radius = fft_a.shape[0] // 8
    log_a = log_polar_transform(fft_a, radius)
    log_b = log_polar_transform(fft_b, radius)
    print(f"  > Log Polar Transform took {time.time() - time_start} seconds")

    # Apply Phase Correlation
    time_start = time.time()
    shifts, error, phasediff = phase_cross_correlation(
        log_a, log_b, upsample_factor=10, normalization=None, disambiguate=True
    )
    print(f"  > Phase Correlation took {time.time() - time_start} seconds")

    # Calculate scale and rotation
    time_start = time.time()
    angle = shifts[0] * 360 / log_a.shape[0]
    klog = a.shape[1] / np.log(radius)
    scale = np.exp(shifts[1] / klog)
    print(f"  > Calculating scale and rotation took {time.time() - time_start} seconds")

    # Create rotation and scaling transformation
    time_start = time.time()
    center = np.array(a.shape) // 2
    rtform = SimilarityTransform(translation=-center)
    rtform += SimilarityTransform(rotation=np.deg2rad(angle))
    rtform += SimilarityTransform(scale=1 if force_scale else scale, translation=center)
    print(f"  > Creating rotation and scaling transformation took {time.time() - time_start} seconds")

    # Apply rotation and scaling
    time_start = time.time()
    rotated_b = warp(b.copy(), rtform.inverse, output_shape=a.shape)
    rotated_mask_b = warp(mask.copy(), rtform.inverse, output_shape=a.shape)
    print(f"  > Applying rotation and scaling took {time.time() - time_start} seconds")

    # Band-pass filter rotated image
    time_start = time.time()
    blurred_rotated_b = difference_of_gaussians(rotated_b, 5, 20)
    print(f"  > Band-pass filtering rotated image took {time.time() - time_start} seconds")

    # Prepare and apply mask
    time_start = time.time()
    # Add black border to the mask
    rotated_mask_b = np.pad(rotated_mask_b, 100, mode='constant', constant_values=0)
    # Resize mask to image size
    rotated_mask_b = resize(rotated_mask_b, rotated_b.shape, anti_aliasing=True)
    # Gaussian blur the mask
    rotated_mask_b = gaussian_filter(rotated_mask_b, sigma=10)
    # Apply mask
    rotated_masked_b = blurred_rotated_b * rotated_mask_b
    print(f"  > Masking rotated image took {time.time() - time_start} seconds")

    # Apply Phase Correlation
    time_start = time.time()
    shifts, error, phasediff = phase_cross_correlation(
        masked_a, rotated_masked_b, upsample_factor=10, normalization=None, disambiguate=True
    )
    print(f"  > Phase Correlation for masked images took {time.time() - time_start} seconds")

    # Create translation transformation
    time_start = time.time()
    ttform = SimilarityTransform(translation=[shifts[1], shifts[0]], scale=1 if force_scale else scale)
    print(f"  > Creating translation transformation took {time.time() - time_start} seconds")

    # Combine transformations
    time_start = time.time()
    ctform = rtform + ttform
    print(f"  > Combining transformations took {time.time() - time_start} seconds")

    return ctform

