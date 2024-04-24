import sys
import gzip
import google.protobuf
import delimited_protobuf
from protocol import message_formats_pb2, telemetry_pb2
import numpy as np


def read_binlog(file_path):
    with gzip.open(file_path, "rb") as fin:
        while True:
            try:
                binlog = delimited_protobuf.read(fin, message_formats_pb2.BinlogRecord)
                if (
                        telemetry_pb2.OculusPingTel.DESCRIPTOR.full_name
                        not in binlog.payload.type_url
                ):
                    continue
                ping = telemetry_pb2.OculusPingTel()
                ping.ParseFromString(binlog.payload.value)
                yield ping.ping, binlog.clock_monotonic.seconds + binlog.clock_monotonic.nanos / 1e9
            except Exception as e:
                print(e)
                return


def read_ping(file_path, start_frame=0, max_frames=None):
    n_frames = 0
    for ping, ts in read_binlog(file_path):
        if ping.ping_id < start_frame:
            continue
        if max_frames is not None and n_frames > max_frames:
            return

        # Convert the string into an array
        data_array = np.frombuffer(ping.ping_data, dtype=np.uint8)

        # Reshape the array into a 2D array
        image_array = data_array.reshape(
            ping.number_of_ranges, ping.number_of_beams + 4
        )

        # Separate gain (first 4 bytes) and data
        gain = np.sqrt(image_array[:, :4].view(np.uint32))
        data = image_array[:, 4:]

        # Calculate the aperture
        aperture = (ping.bearings[-1] - ping.bearings[0]) / 100.0

        n_frames += 1
        yield ping.ping_id, data, gain, aperture, ts


if __name__ == "__main__":
    # Take path from command line
    path = sys.argv[1] if len(sys.argv) > 1 else "../logs/log-multibeam.bez"

    # Open generator
    generator = read_ping(path)

    # Read the first ping
    ping_id, ping_data, ping_gain, ping_aperture, ping_ts = next(generator)

    count = 0
    start_time = last_time = ping_ts
    for ping_id, ping_data, ping_gain, ping_aperture, ping_ts in generator:
        last_time = ping_ts
        count += 1

    print(f"Total pings: {count}")
    print(f"Start time: {start_time}s")
    print(f"End time: {last_time}s")
    print(f"Duration: {last_time - start_time}s")
