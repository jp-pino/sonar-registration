import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import warp

from utils import binlog, image
from registration.fourier_mellin import fourier_mellin
from registration.fourier_mellin_gpu import fourier_mellin as fourier_mellin_gpu

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Take path from command line
    path = sys.argv[1] if len(sys.argv) > 1 else "./logs/log-multibeam.bez"
    out = sys.argv[2] if len(sys.argv) > 2 else "./out"
    start_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # make sure the output directory exists
    os.makedirs(out, exist_ok=True)

    # Open generator
    generator = binlog.read_ping(path, start_frame=start_frame)

    # Read the first ping
    ping_id, data, gain, aperture, ts = next(generator)

    ping_count = 0
    start_time = last_time = ts
    combined = None
    total_tform = None
    a_id, a_raw, _, a_aperture, a_ts = next(generator)
    while True:
        start_time = time.time()
        b_id, b_raw, _, b_aperture, b_ts = next(generator)

        print(f"Processing pings {a_id} and {b_id}")
        print(f"  > Time reading pings: {time.time() - start_time} seconds")

        start = time.time()

        # # Apply fan transformation
        # a, a_mask = image.extract_data_and_mask(image.to_fan(a_raw, a_aperture))
        # b, b_mask = image.extract_data_and_mask(image.to_fan(b_raw, b_aperture))
        a = a_raw
        b = b_raw


        # Mask should be the same size as the image with a black border of 5% of the image size
        mask = np.ones_like(a)
        mask[:int(mask.shape[0] * 0.05), :] = 0
        mask[-int(mask.shape[0] * 0.05):, :] = 0
        mask[:, :int(mask.shape[1] * 0.05)] = 0
        mask[:, -int(mask.shape[1] * 0.05):] = 0

        # Apply gaussian blur to the mask
        mask = np.

        # Pad the images by half of the image size
        a = np.pad(a, np.max(a.shape) // 2)
        b = np.pad(b, np.max(b.shape) // 2)
        mask = np.pad(a_mask, np.max(a_mask.shape) // 2)
        print(f"  > Fan transformation and padding took {time.time() - start} seconds")

        # plt.imsave(os.path.join(out, f"frame_{a_id:08}.png"), a, cmap="gray")
        # plt.imsave(os.path.join(out, f"frame_{b_id:08}.png"), b, cmap="gray")
        normalized_mask = mask / np.max(mask)

        if combined is None:
            masked_a = a * normalized_mask
            combined = masked_a / np.max(masked_a)

        # Apply Fourier-Mellin transformation (and time it)
        start = time.time()
        tform = fourier_mellin(a, b, mask.copy())
        if total_tform is None:
            total_tform = tform
        else:
            total_tform += tform
        print(f"  > Fourier-Mellin transformation took {time.time() - start} seconds")

        start = time.time()
        masked_b = b * normalized_mask
        b = warp(masked_b, total_tform.inverse, output_shape=b.shape)


        # Overlay the images
        combined = np.maximum(combined, b / np.max(b))
        print(f"  > Image overlay took {time.time() - start} seconds")

        plt.imsave(os.path.join(out, f"combined_{b_id:08}.png"), combined, cmap="gray")

        a_id, a_raw, _, a_aperture, a_ts = b_id, b_raw, _, b_aperture, b_ts
