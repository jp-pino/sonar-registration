import sys
import gzip
import google.protobuf
import delimited_protobuf
from protocol import message_formats_pb2, telemetry_pb2
import numpy as np


def read_binlog(file_path, message=telemetry_pb2.MultibeamPingTel):
    with gzip.open(file_path, "rb") as fin:
        while True:
            try:
                binlog = delimited_protobuf.read(fin, message_formats_pb2.BinlogRecord)
                if (
                        message.DESCRIPTOR.full_name
                        not in binlog.payload.type_url
                ):
                    continue
                ping = message()
                ping.ParseFromString(binlog.payload.value)
                yield ping, binlog.clock_monotonic.seconds + binlog.clock_monotonic.nanos / 1e9
            except Exception as e:
                print(e)
                return


def read_ping(file_path, start_frame=0, max_frames=None):
    n_frames = 0
    for ping, data, ts in read_ping_2(file_path, start_frame, max_frames):
        n_frames += 1

        # Calculate the aperture
        fov = np.max(ping.bearings) - np.min(ping.bearings)
        print(f"Max bearing: {np.max(ping.bearings)}")
        print(f"Min bearing: {np.min(ping.bearings)}")
        print(f"FoV: {fov}")

        yield n_frames, data, None, fov, ping.range / ping.number_of_ranges, ts


def read_ping_2(file_path, start_frame=0, max_frames=None):
    ping_generator = read_binlog(file_path)
    n_frames = 0

    for i in range(start_frame):
        next(ping_generator)

    for ping, ts in ping_generator:
        n_frames += 1
        print("Frame", n_frames)
        if max_frames is not None and n_frames > max_frames:
            return

        ping = ping.ping

        # Convert the string into an array
        raw = np.frombuffer(ping.ping_data, dtype=np.uint8)

        # Reshape the array into a 2D array
        raw = raw.reshape(
            ping.number_of_ranges, ping.number_of_beams
        )

        yield ping, raw, ts


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
