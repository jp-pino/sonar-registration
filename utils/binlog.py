import sys
import gzip
import google.protobuf
import delimited_protobuf
from protocol import message_formats_pb2, telemetry_pb2
import numpy as np


def read_binlog_2(file_path, type_names=[telemetry_pb2.MultibeamPingTel]):
    with gzip.open(file_path, "rb") as fin:
        while True:
            binlog = delimited_protobuf.read(fin, message_formats_pb2.BinlogRecord)
            if not binlog:
                break
            type_name = next((type_name for type_name in type_names if
                              type_name.DESCRIPTOR.full_name in binlog.payload.type_url), None)
            if not type_name:
                continue
            message = type_name()
            message.ParseFromString(binlog.payload.value)
            yield message, binlog.clock_monotonic.seconds + binlog.clock_monotonic.nanos / 1e9


# All messages will be yielded in chronological order
def read_combined(file_paths, type_names=[telemetry_pb2.MultibeamPingTel]):
    generators = [read_binlog_2(file_path, type_names) for file_path in file_paths]
    buffer = []
    remove = []

    start_ts = None
    interval = 1

    if telemetry_pb2.MultibeamPingTel in type_names:
        print("Searching for start timestamp")
        iteration = 0
        while not start_ts and generators and iteration < 5:
            for i, generator in enumerate(generators):
                try:
                    message, ts = next(generator)
                except StopIteration:
                    remove.append(i)
                    break

                buffer.append((message, ts))

                if isinstance(message, telemetry_pb2.MultibeamPingTel):
                    start_ts = ts
                    print(f"Start timestamp: {start_ts}")
                    break

            for i in remove:
                generators.pop(i)
            remove = []

            iteration += 1

        if start_ts:
            buffer = [x for x in buffer if x[1] >= start_ts]

    while generators:
        for i, generator in enumerate(generators):
            ts = start_ts
            count = 0
            while ts - start_ts < interval:
                try:
                    message, ts = next(generator)
                    if ts > start_ts:
                        buffer.append((message, ts))
                    count += 1
                except StopIteration:
                    remove.append(i)
                    break

        for i in remove:
            generators.pop(i)
        remove = []

        buffer.sort(key=lambda x: x[1])

        for message, ts in buffer:
            if start_ts <= ts < start_ts + interval:
                yield message, ts

        start_ts += interval

    return


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


def read_ping_2(file_paths, start_frame=0, max_frames=None):
    ping_generator = read_combined(file_paths)
    n_frames = 0

    for i in range(start_frame):
        next(ping_generator)

    for ping, ts in ping_generator:
        n_frames += 1
        print("Frame", n_frames)
        if max_frames is not None and n_frames > max_frames:
            return

        ping, raw = process_ping(ping)

        yield ping, raw, ts


def process_ping(msg):
    ping = msg.ping

    if hasattr(msg, 'raw'):
        raw = msg.raw
        return ping, raw

    # Convert the string into an array
    raw = np.frombuffer(ping.ping_data, dtype=np.uint8)

    # Reshape the array into a 2D array
    raw = raw.reshape(
        ping.number_of_ranges, ping.number_of_beams
    )

    return ping, raw


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
