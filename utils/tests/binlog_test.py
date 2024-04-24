import unittest
from utils import binlog

class TestBinlog(unittest.TestCase):
    def test_read(self):

        # Open generator
        generator = binlog.read_ping("../../logs/log-multibeam.bez")

        # Read the first ping
        ping_id, ping_data, ping_gain, ping_aperture, ping_ts = next(generator)

        count = 1
        start_time = last_time = ping_ts
        for ping_id, ping_data, ping_gain, ping_aperture, ping_ts in generator:
            last_time = ping_ts
            count += 1

        self.assertEqual(count, 4039)
        self.assertEqual(start_time, 12606.097112828)
        self.assertEqual(last_time, 13333.251555247)