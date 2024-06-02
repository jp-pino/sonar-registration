import ray

from pipeline.PipelineModule import PipelineModule

from utils import image

from colorama import Fore, Style
import numpy as np

from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
from skimage.registration import phase_cross_correlation
from skimage.transform import (
    warp_polar,
    resize,
    SimilarityTransform,
    warp,
)


class IdentityModule(PipelineModule):
    def run(self, a, b, mask, tform):
        return a, b, mask, tform


class ResizeModule(PipelineModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    @staticmethod
    @ray.remote
    def resize(img, size):
        if img is None:
            return None

        total_size = img.shape[0] * img.shape[1]

        if total_size <= size:
            print(f"    > {Fore.YELLOW}Warning: Image size ({total_size}) is smaller than target size({size}), skipping resizing{Style.RESET_ALL}")
            return img

        ratio = size / total_size
        img = resize(img.copy(), (int(img.shape[0] * ratio), int(img.shape[1] * ratio)), anti_aliasing=True)
        return img

    def run(self, a, b, mask, tform):
        a = self.resize.remote(a, self.size)
        b = self.resize.remote(b, self.size)
        mask = self.resize.remote(mask, self.size)

        return ray.get(a), ray.get(b), ray.get(mask), tform


class FanModule(PipelineModule):
    def __init__(self, aperture):
        super().__init__()
        self.aperture = aperture

    @staticmethod
    @ray.remote
    def get_masked_fan(raw, aperture):
        if raw is None:
            return None, None
        data, mask = image.extract_data_and_mask(image.to_fan(raw, aperture))
        mask = np.interp(mask, (mask.min(), mask.max()), (0, 1))
        return data * mask, mask

    def run(self, a, b, mask, tform):
        a = self.get_masked_fan.remote(a, self.aperture)
        b = self.get_masked_fan.remote(b, self.aperture)
        mask = self.get_masked_fan.remote(mask, self.aperture)

        a, _ = ray.get(a)
        b, _ = ray.get(b)
        mask, _ = ray.get(mask)

        return a, b, mask, tform


class PaddingModule(PipelineModule):
    def __init__(self, padding_ratio):
        super().__init__()
        self.padding_ratio = padding_ratio

    @staticmethod
    @ray.remote
    def pad(data, pad_size):
        if data is None:
            return None
        return np.pad(data, pad_size, mode='constant', constant_values=0)

    def run(self, a, b, mask, tform):
        pad_size = np.max(a.shape) // self.padding_ratio
        a = self.pad.remote(a, pad_size)
        b = self.pad.remote(b, pad_size)
        mask = self.pad.remote(mask, pad_size)

        return ray.get(a), ray.get(b), ray.get(mask), tform


class BandpassModule(PipelineModule):
    def __init__(self, low_cutoff, high_cutoff):
        super().__init__()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    @staticmethod
    @ray.remote
    def bandpass(data, low_cutoff, high_cutoff):
        if data is None:
            return None
        return difference_of_gaussians(data, low_cutoff, high_cutoff)

    def run(self, a, b, mask, tform):
        a = self.bandpass.remote(a, self.low_cutoff, self.high_cutoff)
        b = self.bandpass.remote(b, self.low_cutoff, self.high_cutoff)
        return ray.get(a), ray.get(b), mask, tform


class MaskModule(PipelineModule):
    def __init__(self, sigma=10, padding=150):
        super().__init__()
        self.sigma = sigma
        self.padding = padding
        self.mask = None

    @staticmethod
    @ray.remote
    def apply_mask(data, mask):
        if data is None:
            return None
        return data * mask

    def run(self, a, b, mask, tform):
        if self.mask is None:
            self.mask = np.pad(mask, self.padding, mode='constant', constant_values=0)
            # Resize mask to image size
            self.mask = resize(self.mask, mask.shape, anti_aliasing=True)
            # Gaussian blur the mask
            self.mask = gaussian_filter(self.mask, sigma=self.sigma)

        a = self.apply_mask.remote(a, self.mask)
        b = self.apply_mask.remote(b, self.mask)

        return ray.get(a), ray.get(b), mask, tform


class FourierModule(PipelineModule):
    @staticmethod
    @ray.remote
    def fft(data):
        if data is None:
            return None
        return np.abs(np.fft.fftshift(np.fft.fft2(data)))

    def run(self, a, b, mask, tform):
        a = self.fft.remote(a)
        b = self.fft.remote(b)

        return ray.get(a), ray.get(b), mask, tform


class LogPolarModule(PipelineModule):
    @staticmethod
    @ray.remote
    def log_polar_transform(data, radius, order):
        if data is None:
            return None
        shape = data.shape
        return warp_polar(data, radius=radius, scaling='log', order=order, output_shape=shape)

    def __init__(self, radius_factor=8, order=3):
        super().__init__()
        self.radius_factor = radius_factor
        self.order = order
        self.radius = None

    def run(self, a, b, mask, tform):
        self.radius = a.shape[0] // self.radius_factor

        a = self.log_polar_transform.remote(a, self.radius, self.order)
        b = self.log_polar_transform.remote(b, self.radius, self.order)

        return ray.get(a), ray.get(b), mask, tform


class PhaseCorrelationModule(PipelineModule):
    def __init__(self, upsample_factor=10, mode='rotation', normalization=None, force_scale=True, invert=False):
        super().__init__()
        if mode not in ['rotation', 'translation']:
            raise ValueError("mode must be either 'rotation' or 'translation'")
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.force_scale = force_scale
        self.invert = invert
        self.normalization = normalization

    def run(self, a, b, mask, tform):
        if a is None or b is None:
            raise ValueError("Both images must be provided")

        center = np.array(a.shape) // 2

        shifts, error, phasediff = phase_cross_correlation(
            a, b, upsample_factor=self.upsample_factor, normalization=self.normalization, disambiguate=False
        )
        print(f"    > Shifts: {shifts}, Error: {error}, Phasediff: {phasediff}")

        if self.mode == 'rotation':
            angle = shifts[0] * 360 / a.shape[0]
            if self.invert:
                angle = -angle
            radius = self.find_root().get_module_by_type(LogPolarModule.__name__)[0].radius
            klog = a.shape[1] / np.log(radius)
            print(f"    > Radius: {radius}")
            scale = np.exp(shifts[1] / klog)
            print(f"    > Angle: {angle}, Scale: {scale}")
            tform += SimilarityTransform(translation=-center)
            tform += SimilarityTransform(scale=1 if self.force_scale else scale, rotation=np.deg2rad(angle))
            tform += SimilarityTransform(translation=center)
        else:
            if self.invert:
                shifts = -shifts
            tform += SimilarityTransform(translation=[shifts[1], shifts[0]])
            print(f"    > Translation: {shifts}")

        return a, b, mask, tform


class WarpModule(PipelineModule):
    def __init__(self, combine=False, use_total_transform=False):
        super().__init__()
        self.combine = combine
        self.use_total_transform = use_total_transform

    def run(self, a, b, mask, tform):
        if self.find_root().combined is None:
            self.find_root().combined = np.zeros_like(a if a is not None else b)

        if self.combine:
            b = self.pad_and_combine(b, mask) if b is not None else None
        else:
            t = self.find_root().total_tform if self.use_total_transform else tform
            a = warp(a, t, output_shape=a.shape) if a is not None else None
            b = warp(b, t.inverse, output_shape=b.shape) if b is not None else None

        return a, b, mask, tform

    def pad_and_combine(self, img, mask):
        # Find necessary padding for combining warped image
        margin = np.max(img.shape) // 5
        source_corners = np.array([[margin, margin], [margin, img.shape[0] - margin], [img.shape[1] - margin, img.shape[0] - margin], [img.shape[1] - margin, margin]])
        corners = (self.find_root().centering_tform.inverse + self.find_root().total_tform)(source_corners)

        max_x = np.max(corners[:, 0])
        max_y = np.max(corners[:, 1])
        min_x = np.min(corners[:, 0])
        min_y = np.min(corners[:, 1])

        left = int(np.abs(min_x)) if min_x < 0 else 0
        right = int(max_x - self.find_root().combined.shape[1] - 1) if max_x >= self.find_root().combined.shape[1] else 0
        top = int(np.abs(min_y)) if min_y < 0 else 0
        bottom = int(max_y - self.find_root().combined.shape[0] - 1) if max_y >= self.find_root().combined.shape[0] else 0

        # Update centering transform
        self.find_root().centering_tform += SimilarityTransform(translation=(-left, -top))

        # Pad combined image to fit new corners
        self.find_root().combined = np.pad(self.find_root().combined, ((top, bottom), (left, right)),
                                        mode='constant', constant_values=0)

        print(f"    > Image size: {img.shape}, Mask size: {mask.shape}")
        img[mask != 1] = np.nan
        img = warp(img, (self.find_root().centering_tform + self.find_root().total_tform.inverse), output_shape=self.find_root().combined.shape)
        # w_a = 1.0 * self.find_root().combined_count / (self.find_root().combined_count + 1)
        # w_b = 1.0 / (self.find_root().combined_count + 1)
        # print(f"    > Combining with weights {w_a} and {w_b}")
        # self.find_root().combined = np.nanmean(np.dstack((self.find_root().combined * w_a, img * w_b)), axis=2)
        # self.find_root().combined = np.nanmean(np.dstack((self.find_root().combined, img)), axis=2)
        self.find_root().combined = np.nansum(np.dstack((self.find_root().combined * (self.find_root().combined_count / (self.find_root().combined_count + 1)), (img / (self.find_root().combined_count + 1)))), 2)

        self.find_root().combined_count += 1
        # self.find_root().combined = np.maximum(self.find_root().combined, img)
        self.find_root().combined[np.isnan(self.find_root().combined)] = 0
        img[np.isnan(img)] = 0
        return img
