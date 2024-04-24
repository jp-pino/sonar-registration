import numpy as np
from wand.color import Color
from wand.image import Image
from skimage import exposure


def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    equalized = np.interp(img, bin_centers, img_cdf)
    return np.interp(equalized, (equalized.min(), equalized.max()), (0, 255)).astype(np.uint32)


def to_fan(data, aperture):
    img = Image.from_array(data)
    img.background_color = Color("transparent")
    img.virtual_pixel = "background"
    args = (aperture, 0, img.height, 0)  # ArcAngle  # RotateAngle
    img.flip()
    img.distort("arc", args)
    return img


def extract_data_and_mask(img):
    img = np.array(img)
    data = img[:, :, 0]
    mask = img[:, :, 1]
    return data, mask

