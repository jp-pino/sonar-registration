import unittest

import matplotlib.pyplot as plt

from utils import binlog, image


class TestFan(unittest.TestCase):
    def test_fan(self):
        # Open generator
        generator = binlog.read_ping("../../logs/log-multibeam.bez")

        # Skip the first 100 pings
        for _ in range(100):
            next(generator)

        # Read the first ping
        _, data, _, aperture, ts = next(generator)

        img = image.to_fan(data, aperture)
        img.save(filename="out/fan.png")
        data, mask = image.extract_data_and_mask(img)
        plt.imsave("out/fan_data.png", data, cmap="gray")
        plt.imsave("out/mask_data.png", mask, cmap="gray")

        self.assertTrue(True)
