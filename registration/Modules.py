from registration.PipelineModule import PipelineModule
from registration.Pipeline import Pipeline

from utils import image

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


class FanModule(PipelineModule):
    def __init__(self, aperture):
        super().__init__()
        self.aperture = aperture

    @staticmethod
    def get_masked_fan(raw, aperture):
        data, mask = image.extract_data_and_mask(image.to_fan(raw, aperture))
        mask = np.interp(mask, (mask.min(), mask.max()), (0, 1))
        return data * mask, mask

    def run(self, a, b, mask, tform):
        if a is not None:
            a, mask = self.get_masked_fan(a, self.aperture)
        if b is not None:
            b, mask = self.get_masked_fan(b, self.aperture)

        return a, b, mask, tform


class PaddingModule(PipelineModule):
    def __init__(self, padding_ratio):
        super().__init__()
        self.padding_ratio = padding_ratio
        self.center = None

    def run(self, a, b, mask, tform):
        pad_size = np.max(a.shape) // self.padding_ratio
        if a is not None:
            a = np.pad(a, pad_size, mode='constant', constant_values=0)
        if b is not None:
            b = np.pad(b, pad_size, mode='constant', constant_values=0)
        mask = np.pad(mask, pad_size, mode='constant', constant_values=0)

        if self.center is None:
            self.center = np.array(a.shape) // 2

        return a, b, mask, tform


class BandpassModule(PipelineModule):
    def __init__(self, low_cutoff, high_cutoff):
        super().__init__()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def run(self, a, b, mask, tform):
        if a is not None:
            a = difference_of_gaussians(a, self.low_cutoff, self.high_cutoff)
        if b is not None:
            b = difference_of_gaussians(b, self.low_cutoff, self.high_cutoff)

        return a, b, mask, tform


class MaskModule(PipelineModule):
    def __init__(self, sigma=10, padding=150):
        super().__init__()
        self.sigma = sigma
        self.padding = padding
        self.mask = None

    def run(self, a, b, mask, tform):
        if self.mask is None:
            self.mask = np.pad(mask.copy(), self.padding, mode='constant', constant_values=0)
            # Resize mask to image size
            self.mask = resize(self.mask, mask.shape, anti_aliasing=True)
            # Gaussian blur the mask
            self.mask = gaussian_filter(self.mask, sigma=self.sigma)

        if a is not None:
            a *= self.mask
        if b is not None:
            b *= self.mask

        return a, b, mask, tform


class FourierModule(PipelineModule):
    @staticmethod
    def fft(data):
        return np.fft.fftshift(np.fft.fft2(data))

    def run(self, a, b, mask, tform):
        if a is not None:
            a = np.abs(self.fft(a))
        if b is not None:
            b = np.abs(self.fft(b))

        return a, b, mask, tform


class LogPolarModule(PipelineModule):
    @staticmethod
    def log_polar_transform(data, radius=None, order=None):
        if radius is None:
            radius = data.shape[0] // 2
        shape = data.shape
        return warp_polar(data, radius=radius, scaling='log', order=order, output_shape=shape)

    def __init__(self, radius_factor=8):
        super().__init__()
        self.radius_factor = radius_factor
        self.radius = None

    def run(self, a, b, mask, tform):
        self.radius = a.shape[0] // self.radius_factor

        if a is not None:
            a = self.log_polar_transform(a, self.radius, 3)
        if b is not None:
            b = self.log_polar_transform(b, self.radius, 3)

        return a, b, mask, tform


class PhaseCorrelationModule(PipelineModule):
    def __init__(self, upsample_factor=10, mode='rotation', force_scale=True):
        super().__init__()
        if mode not in ['rotation', 'translation']:
            raise ValueError("mode must be either 'rotation' or 'translation'")
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.force_scale = force_scale

    def run(self, a, b, mask, tform):
        if a is None or b is None:
            raise ValueError("Both images must be provided")

        center = self.pipeline.get_modules(PaddingModule.__name__)[0].center \
            if len(self.pipeline.get_modules(PaddingModule.__name__)) > 0 \
            else np.array(a.shape) // 2

        shifts, error, phasediff = phase_cross_correlation(
            a, b, upsample_factor=self.upsample_factor, normalization=None, disambiguate=True
        )

        if self.mode == 'rotation':
            angle = shifts[0] * 360 / a.shape[0]
            radius = self.pipeline.get_modules(LogPolarModule.__name__)[0].radius
            klog = a.shape[1] / np.log(radius)
            print(f"    > Radius: {radius}")
            scale = np.exp(shifts[1] / klog)
            print(f"    > Angle: {angle}, Scale: {scale}")
            tform += SimilarityTransform(translation=-center)
            tform += SimilarityTransform(scale=1 if self.force_scale else scale, rotation=np.deg2rad(angle))
            tform += SimilarityTransform(translation=center)
        else:
            tform += SimilarityTransform(translation=[shifts[1], shifts[0]])
            print(f"    > Translation: {shifts}")

        return a, b, mask, tform


class WarpModule(PipelineModule):
    def __init__(self, combine=False, use_total_transform=False):
        super().__init__()
        self.combine = combine
        self.use_total_transform = use_total_transform

    def run(self, a, b, mask, tform):
        if self.pipeline.combined is None:
            self.pipeline.combined = np.zeros_like(a if a is not None else b)

        if self.combine:
            b = self.pad_and_combine(b) if b is not None else None
        else:
            t = self.pipeline.total_tform if self.use_total_transform else tform
            a = warp(a, t, output_shape=a.shape) if a is not None else None
            b = warp(b, t.inverse, output_shape=b.shape) if b is not None else None

        return a, b, mask, tform

    def pad_and_combine(self, img):
        # Find necessary padding for combining warped image
        margin = np.max(img.shape) // 5
        source_corners = np.array([[margin, margin], [margin, img.shape[0] - margin], [img.shape[1] - margin, img.shape[0] - margin], [img.shape[1] - margin, margin]])
        corners = (self.pipeline.centering_tform.inverse + self.pipeline.total_tform)(source_corners)

        max_x = np.max(corners[:, 0])
        max_y = np.max(corners[:, 1])
        min_x = np.min(corners[:, 0])
        min_y = np.min(corners[:, 1])

        print(f"    > Max X: {max_x}, Max Y: {max_y}, Min X: {min_x}, Min Y: {min_y}")
        print(f"    > Image Shape: {self.pipeline.combined.shape}")
        left = int(np.abs(min_x)) if min_x < 0 else 0
        right = int(max_x - self.pipeline.combined.shape[1] - 1) if max_x >= self.pipeline.combined.shape[1] else 0
        top = int(np.abs(min_y)) if min_y < 0 else 0
        bottom = int(max_y - self.pipeline.combined.shape[0] - 1) if max_y >= self.pipeline.combined.shape[0] else 0

        print(f"    > Padding (L R T B): {left}, {right}, {top}, {bottom}")

        # left = 2
        # right = 1
        # top = bottom = 4
        # if left > 0 or right > 0 or top > 0 or bottom > 0:
        self.pipeline.padding_left += left
        self.pipeline.padding_right += right
        self.pipeline.padding_top += top
        self.pipeline.padding_bottom += bottom
        print(f"    > Total Padding (L R T B): {self.pipeline.padding_left}, {self.pipeline.padding_right}, {self.pipeline.padding_top}, {self.pipeline.padding_bottom}")

        # Update centering transform
        self.pipeline.centering_tform += SimilarityTransform(translation=(-left, -top))
        print(f"    > Centering TFORM: {self.pipeline.centering_tform.translation}")

        # Pad combined image to fit new corners
        self.pipeline.combined = np.pad(self.pipeline.combined, ((top, bottom), (left, right)),
                                        mode='constant', constant_values=0)

        img = warp(img, (self.pipeline.centering_tform + self.pipeline.total_tform.inverse), output_shape=self.pipeline.combined.shape)

        # self.pipeline.combined = self.pipeline.combined * self.pipeline.combined_count + img
        # self.pipeline.combined_count += 1
        # self.pipeline.combined /= self.pipeline.combined_count
        self.pipeline.combined = np.maximum(self.pipeline.combined, img)
        return img


class UpdateTformModule(PipelineModule):
    def run(self, a, b, mask, tform):
        self.pipeline.total_tform += tform
        return a, b, mask, tform
