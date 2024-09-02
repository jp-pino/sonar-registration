import os

import ray
from matplotlib import pyplot as plt
import scipy.ndimage as ndi

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

from skimage import metrics


class MetricsModule(PipelineModule):
    def __init__(self, output=None):
        super().__init__()
        self.psnr = []
        self.ssim = []
        self.output = output

    def __del__(self):
        if self.output is not None and len(self.psnr) > 0:
            print(f"    > Saving metrics to {self.output}")
            np.savetxt(os.path.join(self.output, "metrics.csv"), np.array([self.psnr, self.ssim]).T, delimiter=",")

    def run(self, a, b, mask, tform, error):
        gaussian_image = gaussian_filter(b, sigma=1)

        print(f"min: {np.min(b)}, max: {np.max(b)}")
        print(f"min: {np.min(gaussian_image)}, max: {np.max(gaussian_image)}")

        self.psnr.append(metrics.peak_signal_noise_ratio(b, gaussian_image))
        self.ssim.append(metrics.structural_similarity(b, gaussian_image, data_range=1))

        # Print nupmy type of the image
        print(f"    > PSNR: {self.psnr[-1]}")
        print(f"    > SSIM: {self.ssim[-1]}")

        return a, b, mask, tform, error


class IdentityModule(PipelineModule):
    def run(self, a, b, mask, tform, error):
        return a, b, mask, tform, error


class RemapModule(PipelineModule):
    def __init__(self, original_range, target_range):
        super().__init__()
        self.original_range = original_range
        self.target_range = target_range

    @staticmethod
    @ray.remote
    def remap(data, original_range, target_range):
        if data is None:
            return None
        return data * target_range / original_range

    def run(self, a, b, mask, tform, error):
        a = self.remap.remote(a, self.original_range, self.target_range)
        b = self.remap.remote(b, self.original_range, self.target_range)
        return ray.get(a), ray.get(b), mask, tform, error


class ResizeModule(PipelineModule):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = np.sqrt(ratio)
        self.init_origin = False

    @staticmethod
    @ray.remote
    def resize(img, ratio):
        if img is None:
            return None
        img = resize(img.copy(), (int(img.shape[0] * ratio), int(img.shape[1] * ratio)), anti_aliasing=True)
        return img

    def run(self, a, b, mask, tform, error):
        if not self.init_origin:
            total_size = a.shape[0] * a.shape[1]
            self.find_root().origin = np.array(a.shape) * self.ratio
            self.init_origin = True

        a = self.resize.remote(a, self.ratio)
        b = self.resize.remote(b, self.ratio)
        mask = self.resize.remote(mask, self.ratio)

        return ray.get(a), ray.get(b), ray.get(mask), tform, error


class BlurModule(PipelineModule):
    def __init__(self, size=3):
        super().__init__()
        self.size = size

    @staticmethod
    @ray.remote
    def blur(img, size):
        if img is None:
            return None
        return ndi.uniform_filter(img, size=size)

    def run(self, a, b, mask, tform, error):
        a = self.blur.remote(a, self.size)
        b = self.blur.remote(b, self.size)
        mask = self.blur.remote(mask, self.size)

        return ray.get(a), ray.get(b), ray.get(mask), tform, error

class FanModule(PipelineModule):
    def __init__(self, fov):
        super().__init__()
        self.fov = fov
        self.init_origin = False

    @staticmethod
    @ray.remote
    def get_masked_fan(raw, fov):
        if raw is None:
            return None, None
        data, mask = image.extract_data_and_mask(image.to_fan(raw, fov))
        mask = np.interp(mask, (mask.min(), mask.max()), (0, 1))
        return data * mask, mask

    def run(self, a, b, mask, tform, error):
        a = self.get_masked_fan.remote(a, self.fov)
        b = self.get_masked_fan.remote(b, self.fov)
        # mask = self.get_masked_fan.remote(mask, self.fov)

        a, _ = ray.get(a)
        b, mask = ray.get(b)
        # mask, _ = ray.get(mask)

        if not self.init_origin:
            self.find_root().origin[1] = a.shape[1] // 2
            self.find_root().origin[0] = a.shape[0]
            self.init_origin = True

        return a, b, mask, tform, error


class FanModule2(PipelineModule):
    def __init__(self, bearings, output=""):
        super().__init__()
        self.bearings = bearings
        self.mapping = None
        self.FOV = None
        self.HEIGHT = None
        self.WIDTH = None
        self.mask = None
        self.init_origin = False
        self.output = output

        # Make sure cache directory exists
        os.makedirs(os.path.join(output, "cache"), exist_ok=True)

    @staticmethod
    @ray.remote
    def get_fan(raw, height, width, mapping):
        if raw is None:
            return None, None

        fan = np.zeros((height, width))
        for x in range(width):
            for y in range(height):
                r, beam_id = mapping[y, x]
                try:
                    if 0 < r < raw.shape[0] and 0 < beam_id < raw.shape[1]:
                        fan[y, x] = raw[r, beam_id]
                except Exception as e:
                    print(f"beam: {beam_id}, r: {r} -> x: {x}, y: {y}")
                    print(f"raw shape: {raw.shape}, fan shape: {fan.shape}")
                    raise e

        return fan

    def get_bearing(self, theta, width):
        index = np.argmin(np.abs(self.bearings - theta))
        return int(index * width / len(self.bearings))

    def run(self, a, b, mask, tform, error):
        FOV = np.max(self.bearings) - np.min(self.bearings)
        HEIGHT = a.shape[0]
        WIDTH = int(2 * HEIGHT * np.sin(np.deg2rad(FOV / 2)))
        mask = None

        print(
            f"    > FOV: {FOV}, HEIGHT: {HEIGHT}, WIDTH: {WIDTH}, Number of bearings: {len(self.bearings)}, RAW Width: {a.shape[1]}, RAW Height: {a.shape[0]}")
        if (self.mapping is None or FOV != self.FOV
                or HEIGHT != self.HEIGHT
                or WIDTH != self.WIDTH):
            self.FOV = FOV
            self.HEIGHT = HEIGHT
            self.WIDTH = WIDTH
            if os.path.exists(os.path.join(self.output, "cache", f"{FOV}_{WIDTH}_{HEIGHT}_mapping.npy")):
                print(f"{Fore.YELLOW}    > Loading mapping from cache")
                self.mapping = np.load(os.path.join(self.output, "cache", f"{FOV}_{WIDTH}_{HEIGHT}_mapping.npy"))
                self.mask = np.load(os.path.join(self.output, "cache", f"{FOV}_{WIDTH}_{HEIGHT}_mask.npy"))
            else:
                print(f"{Fore.YELLOW}    > Recalculating self.mapping")
                print(f"{Fore.YELLOW}    > self.bearings: {self.bearings}")
                print(f"{Fore.YELLOW}    > FOV: {FOV}, HEIGHT: {HEIGHT}, WIDTH: {WIDTH}")
                self.mapping = np.zeros((HEIGHT, WIDTH, 2), dtype=int)
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        theta = np.rad2deg(np.arctan2(x - WIDTH / 2, HEIGHT - y))
                        r = np.sqrt((x - WIDTH / 2) ** 2 + (HEIGHT - y) ** 2)
                        beam_id = -1
                        if np.min(self.bearings) < theta < np.max(self.bearings):
                            beam_id = self.get_bearing(theta, a.shape[1])
                        self.mapping[y, x] = [r, beam_id]
                print(f"    > Mapping shape: {self.mapping.shape}")

                np.save(os.path.join(self.output, "cache", f"{FOV}_{WIDTH}_{HEIGHT}_mapping.npy"), self.mapping)

                self.mask = None
                mask = self.get_fan.remote(np.ones_like(a), self.HEIGHT, self.WIDTH, self.mapping)

                np.save(os.path.join(self.output, "cache", f"{FOV}_{WIDTH}_{HEIGHT}_bearings.npy"), self.bearings)

        a = self.get_fan.remote(a, self.HEIGHT, self.WIDTH, self.mapping)
        b = self.get_fan.remote(b, self.HEIGHT, self.WIDTH, self.mapping)

        a = ray.get(a)
        b = ray.get(b)
        if mask is not None:
            self.mask = ray.get(mask)
            np.save(os.path.join(self.output, "cache", f"{FOV}_{WIDTH}_{HEIGHT}_mask.npy"), self.mask)

        if not self.init_origin:
            self.find_root().origin[1] = a.shape[1] // 2
            self.find_root().origin[0] = a.shape[0]
            self.init_origin = True

        return a, b, self.mask, tform, error


class PaddingModule(PipelineModule):
    def __init__(self, padding_ratio):
        super().__init__()
        self.padding_ratio = padding_ratio
        self.init_origin = False

    @staticmethod
    @ray.remote
    def pad(data, pad_size):
        if data is None:
            return None
        return np.pad(data, pad_size, mode='constant', constant_values=0)

    def run(self, a, b, mask, tform, error):
        pad_size = int(np.max(a.shape) * self.padding_ratio)
        a = self.pad.remote(a, pad_size)
        b = self.pad.remote(b, pad_size)
        mask = self.pad.remote(mask, pad_size)

        if not self.init_origin:
            self.find_root().origin += pad_size
            self.init_origin = True

        return ray.get(a), ray.get(b), ray.get(mask), tform, error


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

    def run(self, a, b, mask, tform, error):
        a = self.bandpass.remote(a, self.low_cutoff, self.high_cutoff)
        b = self.bandpass.remote(b, self.low_cutoff, self.high_cutoff)
        return ray.get(a), ray.get(b), mask, tform, error


class BandpassTestingModule(PipelineModule):
    def __init__(self, output=None):
        super().__init__()
        self.output = output

    @staticmethod
    @ray.remote
    def bandpass(data, low_cutoff, high_cutoff):
        if data is None:
            return None
        return difference_of_gaussians(data, low_cutoff, high_cutoff)

    def run(self, a, b, mask, tform, error):
        for i in range(10, 30):
            for j in range(i - 1):
                plt.imsave(os.path.join(self.output, f"{j}_{i}.png"), difference_of_gaussians(b, j, i))
        return a, b, mask, tform, error


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

    def run(self, a, b, mask, tform, error):
        if self.mask is None:
            self.mask = np.pad(mask, self.padding, mode='constant', constant_values=0)
            # Resize mask to image size
            self.mask = resize(self.mask, mask.shape, anti_aliasing=True)
            # Gaussian blur the mask
            self.mask = gaussian_filter(self.mask, sigma=self.sigma)

        a = self.apply_mask.remote(a, self.mask)
        b = self.apply_mask.remote(b, self.mask)

        return ray.get(a), ray.get(b), self.mask, tform, error


class FourierModule(PipelineModule):
    @staticmethod
    @ray.remote
    def fft(data):
        if data is None:
            return None
        data = np.abs(np.fft.fftshift(np.fft.fft2(data)))
        center = np.array(data.shape) // 2

        # make a black square in the center of the image
        # margin = 0
        # data[center[0] - margin:center[0] + margin, center[1] - margin:center[1] + margin] = 0
        return data


    def run(self, a, b, mask, tform, error):
        a = self.fft.remote(a)
        b = self.fft.remote(b)


        return ray.get(a), ray.get(b), mask, tform, error


class LogPolarModule(PipelineModule):
    @staticmethod
    @ray.remote
    def log_polar_transform(data, radius, order):
        if data is None:
            return None
        shape = data.shape
        return warp_polar(data, radius=radius, scaling='log', order=order, output_shape=shape)

    def __init__(self, radius_factor=1/8, order=3):
        super().__init__()
        self.radius_factor = radius_factor
        self.order = order
        self.radius = None

    def run(self, a, b, mask, tform, error):
        self.radius = int(a.shape[0] * self.radius_factor)

        a = self.log_polar_transform.remote(a, self.radius, self.order)
        b = self.log_polar_transform.remote(b, self.radius, self.order)

        a = ray.get(a)
        b = ray.get(b)

        print(f"    > Radius: {self.radius}")
        print(f"    > Shape: {a.shape}")

        return a, b, mask, tform, error


class PhaseCorrelationModule(PipelineModule):
    def __init__(self, upsample_factor=10, mode='rotation', normalization=None, force_scale=True, log_polar=True,
                 max_rotation=80, max_translation=100):
        super().__init__()
        if mode not in ['rotation', 'translation']:
            raise ValueError("mode must be either 'rotation' or 'translation'")
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.force_scale = force_scale
        self.log_polar = log_polar
        self.normalization = normalization
        self.max_rotation = max_rotation
        self.max_translation = max_translation

    def run(self, a, b, mask, tform, error):
        if a is None or b is None:
            raise ValueError("Both images must be provided")

        center = np.array(a.shape) // 2

        shifts, e, phasediff = phase_cross_correlation(
            a, b, upsample_factor=self.upsample_factor, normalization=self.normalization, disambiguate=False
        )
        print(f"    > Shifts: {shifts}, Error: {e}, Phasediff: {phasediff}")

        if self.mode == 'rotation':
            if self.log_polar:
                angle = shifts[0] * 360 / a.shape[0]
            else:
                angle = shifts[1] * 120 / a.shape[1]

            if np.abs(angle) > self.max_rotation:
                print(f"{Fore.YELLOW}    > Skipping rotation due to high angle: {angle}")
                angle = 0
                error[0] = 1

            if self.log_polar:
                radius = self.find_root().get_module_by_type(LogPolarModule.__name__)[0].radius
                klog = a.shape[1] / np.log(radius)
                scale = np.exp(shifts[1] / klog)
            else:
                scale = 1

            print(f"    > Angle: {angle}, Scale: {scale}")
            tform += SimilarityTransform(translation=-center)
            tform += SimilarityTransform(scale=1 if self.force_scale else scale, rotation=np.deg2rad(angle))
            tform += SimilarityTransform(translation=center)
            error[2] = e if abs(angle) < 60 else 1
        else:
            if np.abs(shifts[0]) > self.max_translation:
                print(f"{Fore.YELLOW}    > Skipping translation due to high shift[0]: {shifts[0]}")
                shifts[0] = 0
                error[0] = 1
            if np.abs(shifts[1]) > self.max_translation:
                print(f"{Fore.YELLOW}    > Skipping translation due to high shift[1]: {shifts[1]}")
                shifts[1] = 0
                error[1] = 1
            tform += SimilarityTransform(translation=[shifts[1], shifts[0]])
            print(f"    > Translation: {shifts}")
            error[0] = e if abs(shifts[0]) < np.max(a.shape) // 2 else 1
            error[1] = e if abs(shifts[0]) < np.max(a.shape) // 2 else 1

        return a, b, mask, tform, error


class WarpModule(PipelineModule):
    def __init__(self, combine=False, use_total_transform=False):
        super().__init__()
        self.combine = combine
        self.use_total_transform = use_total_transform
        self.last_valid = -1
        self.last_total_tform = SimilarityTransform()
        self.count = 0

    def run(self, a, b, mask, tform, error):

        if self.combine:
            if self.find_root().combined is None:
                self.find_root().combined = np.zeros((b.shape[0], b.shape[1]))
            self.count += 1

            current_corners = self.find_corners(self.find_root().total_tform, b.shape, 0)
            last_corners = self.find_corners(self.last_total_tform, b.shape, 0)

            # if np.max(np.abs(current_corners - last_corners)) < 5 and self.last_valid < self.count - 10:
            #     print(f"    > Skipping due to small change in corners")
            #     return a, b, mask, tform, error

            b = self.pad_and_combine(b, mask) if b is not None else None

            if self.last_valid == -1:
                self.last_valid = self.count
                self.last_total_tform = self.find_root().total_tform
        else:
            t = self.find_root().total_tform if self.use_total_transform else tform
            a = warp(a, t, output_shape=a.shape) if a is not None else None
            b = warp(b, t.inverse, output_shape=b.shape) if b is not None else None

        return a, b, mask, tform, error

    @staticmethod
    def find_corners(tform, shape, margin):
        corners = np.array([[margin, margin], [margin, shape[0] - margin], [shape[1] - margin, shape[0] - margin],
                            [shape[1] - margin, margin]])
        return tform(corners)

    def pad_and_combine(self, img, mask):
        # Find necessary padding for combining warped image
        margin = np.max(img.shape) // 10
        # # margin = 5
        source_corners = np.array([[margin, margin], [margin, img.shape[0] - margin], [img.shape[1] - margin, img.shape[0] - margin], [img.shape[1] - margin, margin]])
        corners = (self.find_root().centering_tform + self.find_root().total_tform.inverse).inverse(source_corners)
        tform = self.find_root().centering_tform + self.find_root().total_tform.inverse
        # corners = self.find_corners(tform, img.shape, margin)

        max_x = np.max(corners[:, 0])
        max_y = np.max(corners[:, 1])
        min_x = np.min(corners[:, 0])
        min_y = np.min(corners[:, 1])

        left = int(np.abs(min_x)) if min_x < 0 else 0
        right = int(max_x - self.find_root().combined.shape[1] - 1) if max_x >= self.find_root().combined.shape[
            1] else 0
        top = int(np.abs(min_y)) if min_y < 0 else 0
        bottom = int(max_y - self.find_root().combined.shape[0] - 1) if max_y >= self.find_root().combined.shape[
            0] else 0

        print(f"    > Padding image with L {left}, R {right}, T {top}, B {bottom}")

        # Update centering transform
        self.find_root().centering_tform += SimilarityTransform(translation=(-left, -top))
        print(
            f"    > Centering transform. Translation: {self.find_root().centering_tform.translation} Rotation: {self.find_root().centering_tform.rotation}")

        # Pad combined image to fit new corners
        self.find_root().combined = np.pad(self.find_root().combined, ((top, bottom), (left, right)),
                                           mode='constant', constant_values=0)

        print(f"    > Image size: {img.shape}, Mask size: {mask.shape}")

        # Resize mask
        mask = warp(mask, SimilarityTransform(), output_shape=img.shape)
        # Make values outside of mask np.nan
        img[mask < 1] = np.nan
        # Warp image and make all values outside of image np.nan
        img = warp(img, (self.find_root().centering_tform + self.find_root().total_tform.inverse), output_shape=self.find_root().combined.shape, mode='constant', cval=np.nan)
        # w_a = 1.0 * self.find_root().combined_count / (self.find_root().combined_count + 1)
        # w_b = 1.0 / (self.find_root().combined_count + 1)
        # print(f"    > Combining with weights {w_a} and {w_b}")
        # self.find_root().combined = np.nanmean(np.dstack((self.find_root().combined * w_a, img * w_b)), axis=2)
        # self.find_root().combined = np.nanmean(np.dstack((self.find_root().combined, img)), axis=2)

        previous_combined = self.find_root().combined * (self.find_root().combined_count / (self.find_root().combined_count + 1))
        previous_combined[np.isnan(img)] = self.find_root().combined[np.isnan(img)]
        next_frame = img / (self.find_root().combined_count + 1)

        self.find_root().combined = np.nansum(np.dstack((previous_combined, next_frame)), 2)

        self.find_root().combined_count += 1
        # self.find_root().combined = np.maximum(self.find_root().combined, img)
        self.find_root().combined[np.isnan(self.find_root().combined)] = 0
        img[np.isnan(img)] = 0

        return img
