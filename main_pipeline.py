#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import os
import sys
import signal
import time
from enum import Enum
import argparse
import argcomplete, argparse
from argcomplete.completers import EnvironCompleter

import numpy as np
from matplotlib import pyplot as plt

from pipeline.Pipeline import Pipeline
from protocol import telemetry_pb2
from registration.Modules import *
from alignment.Modules import *
from alignment.plot_slam2d import plot_slam2d

from utils import binlog

EXIT_FLAG = False


def signal_handler(sig, frame):
    global EXIT_FLAG
    EXIT_FLAG = True
    print(f"Signal received: {sig}. Exiting...")
    time.sleep(3)


def get_fake_data(path):
    Object = lambda **kwargs: type("MultibeamPingTel", (), kwargs)
    i = 0
    source = path
    while True:
        path = os.path.join(source, f'raw_{i}.png')
        print(f"Reading {path}")
        try:
            img = plt.imread(path)
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            yield Object(
                ping=Object(bearings=np.array([theta for theta in np.linspace(-60, 60, 520)]), number_of_ranges=520,
                            range=520), raw=img), i * 0.1
            i += 1
        except FileNotFoundError as e:
            print(f"File not found: {path}")
            break
    return


class Algorithm(Enum):
    FOURIER_MELLIN = 1
    PHASE_CORRELATION = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Algorithm[s]
        except KeyError:
            raise ValueError()


def create_fourier_mellin(args, bearings, fov, height, width, range_resolution):
    pipeline = Pipeline(output=args.out, intermediate_output=args.intermediate_output, verbose=args.verbose)
    pipeline.add_module(IdentityModule())

    conditioning_id, conditioning = pipeline.add_module(Pipeline('conditioning'))
    if args.disable_resizing:
        conditioning.add_module(RemapModule(255, 1))
    else:
        conditioning.add_module(ResizeModule(args.resize))
    conditioning.add_module(FanModule2(bearings, output=args.out))
    conditioning.add_module(PaddingModule(4))

    pipeline.add_module(MetricsModule(output=args.out))

    filtering_id, filtering = pipeline.add_module(Pipeline('filtering'))
    # filtering.add_module(BandpassTestingModule(output=out))
    filtering.add_module(BandpassModule(args.bandpass_low, args.bandpass_high))
    filtering.add_module(MaskModule(padding=60, sigma=15))

    registration_id, registration = pipeline.add_module(Pipeline('registration'))
    start_id, _ = registration.add_module(IdentityModule())
    registration.add_module(FourierModule())
    registration.add_module(LogPolarModule(order=1))
    registration.add_module(PhaseCorrelationModule(10, 'rotation'))
    registration.add_module(IdentityModule(), input_stage=start_id)
    registration.add_module(WarpModule(), apply_to=('b', 'm'))
    registration.add_module(PhaseCorrelationModule(10, 'translation'))

    alignment_id, alignment = pipeline.add_module(Pipeline('global_alignment'))
    alignment.add_module(UpdateTformModule(range_resolution=range_resolution))
    if not args.disable_loop_closure:
        alignment.add_module(FindNeighborsModule(registration_id, range_resolution=range_resolution, output=args.out,
                                                 matches_number=args.matches_number,
                                                 error_threshold=args.error_threshold),
                             input_stage=filtering_id)

    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=conditioning_id)
    pipeline.add_module(WarpModule(combine=True), 'b')
    # pipeline.add_module(IdentityModule(), input_stage=filtering_id)
    pipeline.add_module(OdometerModule(output=args.out, range_resolution=range_resolution))

    realignment = Pipeline('realignment')
    realignment.add_module(WarpModule(combine=True), 'b')

    return pipeline, filtering_id, realignment


def create_phase_correlation(args, bearings, fov, height, width, range_resolution):
    pipeline = Pipeline(output=args.out, intermediate_output=args.intermediate_output, verbose=args.verbose)
    pipeline.add_module(MetricsModule(output=args.out))
    registration_id, registration = pipeline.add_module(Pipeline('registration'))

    if not args.disable_resizing:
        registration.add_module(ResizeModule(args.resize))

    resize_id, _ = registration.add_module(IdentityModule())

    if args.disable_resizing:
        registration.add_module(BlurModule(20))

    registration.add_module(MaskModule(padding=60, sigma=15))
    registration.add_module(PhaseCorrelationModule(20, 'rotation', log_polar=False))
    registration.add_module(FanModule2(bearings, output=args.out), input_stage=resize_id)
    padding_id, _ = registration.add_module(PaddingModule(4))
    registration.add_module(BandpassModule(args.bandpass_low, args.bandpass_high))
    registration.add_module(MaskModule(padding=60, sigma=15))
    registration.add_module(WarpModule(), apply_to=('b', 'm'))
    registration.add_module(PhaseCorrelationModule(10, 'translation'))

    pipeline.add_module(UpdateTformModule())
    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=pipeline.name)
    pipeline.add_module(OdometerModule(output=args.out, range_resolution=range_resolution))
    if not args.disable_loop_closure:
        pipeline.add_module(FindNeighborsModule(registration_id, range_resolution=range_resolution, output=args.out,
                                                matches_number=args.matches_number,
                                                error_threshold=args.error_threshold,
                                                delta_theta=(fov / 10), delta_radius=10),
                            input_stage=pipeline.name)
    pipeline.add_module(IdentityModule(), input_stage=padding_id)
    pipeline.add_module(WarpModule(combine=True), 'b')

    realignment = Pipeline("realignment")
    realignment.add_module(ResizeModule(args.resize))
    realignment.add_module(FanModule2(bearings, output=args.out))
    realignment.add_module(PaddingModule(4))
    realignment.add_module(WarpModule(combine=True), 'b')

    return pipeline, registration_id, realignment


# Press the green button in the gutter to run the script.
def main():
    global EXIT_FLAG
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", help="Paths to the log files", nargs='+')
    parser.add_argument("--out", help="Output directory", default="./out")
    parser.add_argument("--skip", help="Number of frames to skip", type=int, default=0)
    parser.add_argument("--output_frequency", help="How many frames to skip before outputting a frame",
                        type=int, default=25)
    parser.add_argument("--redraw_frequency", help="How many frames before output is redrawn",
                        type=int, default=25)
    parser.add_argument("--bandpass_low", help="Low cutoff value for bandpass filter", type=int, default=2)
    parser.add_argument("--bandpass_high", help="High cutoff value for bandpass filter", type=int, default=20)
    parser.add_argument("--matches_number", help="Number of matches to consider", type=int, default=10)
    parser.add_argument("--start_frame", help="Start frame", type=int, default=0)
    parser.add_argument("--max_frames", help="Max frames", type=int, default=None)
    parser.add_argument("--verbose", help="Verbose", action="store_true")
    parser.add_argument("--resize", help="Resize the images", type=int, default=0.25)
    parser.add_argument("--disable_resizing", help="Disable resizing", action="store_true", default=False)
    parser.add_argument("--disable_loop_closure", help="Disable loop closure", action="store_true", default=False)
    parser.add_argument("--disable_realignment", help="Disable realignment", action="store_true", default=False)
    parser.add_argument("--error_threshold", help="Error threshold for loop closure matches", type=float, default=1)
    parser.add_argument("--algorithm", help="Use fourier-mellin transform", choices=list(Algorithm),
                        type=Algorithm.from_string, default=Algorithm.FOURIER_MELLIN)
    parser.add_argument("--intermediate_output", help="Output intermediate images", action="append", type=str)
    parser.add_argument("--print", help="Print the pipeline", action="store_true")
    argcomplete.autocomplete(parser)

    # Parse arguments
    args = parser.parse_args()

    # make sure the output directory exists
    os.makedirs(args.out, exist_ok=True)

    # Open generator
    fake_data = False
    if all(path.endswith(".bez") or path.endswith(".mbez") for path in args.paths):
        print("Reading combined telemetry")
        generator = binlog.read_combined(args.paths, [telemetry_pb2.MultibeamPingTel, telemetry_pb2.PositionEstimateTel,
                                                      telemetry_pb2.CalibratedImuTel])
    else:
        print("Reading fake telemetry")
        fake_data = True
        generator = get_fake_data(args.paths[0])

    # Read the first ping
    i = 0
    msg = None
    while i < args.start_frame:
        msg, a_ts = next(generator)
        if isinstance(msg, telemetry_pb2.MultibeamPingTel) or fake_data:
            i += 1

    if not msg:
        msg, a_ts = next(generator)

    a_ping, a_raw = binlog.process_ping(msg)

    speed_limit = 1.5  # m/s

    bearings = a_ping.bearings
    fov = np.max(bearings) - np.min(bearings)
    height = a_ping.number_of_ranges
    width = int(2 * a_ping.number_of_ranges * np.sin(np.deg2rad(fov / 2)))
    total_size = a_raw.shape[0] * a_raw.shape[1]
    if not args.disable_resizing:
        ratio = np.sqrt(args.resize)
    else:
        ratio = 1
    range_resolution = (a_ping.range / (a_ping.number_of_ranges * ratio))
    print(f"Range resolution: {range_resolution} m")
    print(f"Ping range: {a_ping.range} m")
    print(f"Number of ranges: {a_ping.number_of_ranges}")
    print(f"Total size: {total_size}")
    print(f"Ratio: {ratio}")

    translation_limit = (speed_limit / range_resolution) * 0.1  # pixels / frame

    # print(a_ping.range)
    # print(a_ping.number_of_ranges)
    # print(a_ping.range / a_ping.number_of_ranges, range_resolution)

    if args.algorithm == Algorithm.FOURIER_MELLIN:
        pipeline, redraw_id, realignment = create_fourier_mellin(args, bearings, fov, height, width, range_resolution)
    elif args.algorithm == Algorithm.PHASE_CORRELATION:
        pipeline, redraw_id, realignment = create_phase_correlation(args, bearings, fov, height, width,
                                                                    range_resolution)
    else:
        raise ValueError("Invalid algorithm")

    if args.intermediate_output:
        print(f"Intermediate output enabled: {args.intermediate_output}")

    if args.print:
        return

    count = 1
    prev_northing = prev_easting = start_odometer = 0
    dvl_log = []
    imu_log = []
    while True:
        if EXIT_FLAG:
            break

        try:
            msg, ts = next(generator)
        except StopIteration as e:
            print("End of files")
            break

        if isinstance(msg, telemetry_pb2.MultibeamPingTel) or fake_data:
            if args.max_frames and count > args.max_frames:
                break

            if count % (args.skip + 1) != 0:
                count += 1
                continue

            b_ping, b_raw = binlog.process_ping(msg)
            b_ts = ts

            print(f"\nProcessing pings {count - 1} and {count}")
            pipeline.execute(a_raw, b_raw)

            if pipeline.verbose:
                fig = plot_slam2d(pipeline.pose_graph.optimizer, "Before optimisation")
                fig.write_image(os.path.join(args.out, f"graph_{count}.png"))
                fig.write_html(os.path.join(args.out, f"graph_{count}.html"))

            if not args.disable_realignment:
                pipeline.optimize(100, verbose=args.verbose)

            if pipeline.verbose:
                fig = plot_slam2d(pipeline.pose_graph.optimizer, "After optimisation")
                fig.write_image(os.path.join(args.out, f"graph_{count}_optimized.png"))
                fig.write_html(os.path.join(args.out, f"graph_{count}_optimized.html"))

            if count % args.redraw_frequency == 0 and not args.disable_realignment:
                pipeline.redraw(realignment, redraw_id)

            if count % args.output_frequency == 0:
                plt.imsave(os.path.join(args.out, f"combined_{count:08}.png"), pipeline.combined, cmap="gray")

            # if count % (max_frames / 5 if max_frames is not None else 50) == 0:
            # plt.imsave(os.path.join(args.out, f"combined_{count:08}.png"), pipeline.combined, cmap="gray")
            count += 1
            a_ping, a_raw, a_ts = b_ping, b_raw, b_ts

        elif isinstance(msg, telemetry_pb2.PositionEstimateTel):
            position, pos_ts = msg.position_estimate, ts

            if start_odometer == 0:
                start_odometer = position.odometer

            current_northing = position.northing
            current_easting = position.easting
            current_odometer = position.odometer

            scaled_northing = (current_northing - prev_northing) / range_resolution
            scaled_easting = (current_easting - prev_easting) / range_resolution

            # pipeline.pose_graph.add_landmark_to_last_vertex(scaled_northing, scaled_easting)
            prev_northing = current_northing
            prev_easting = current_easting

            dvl_log.append([count, pos_ts, current_northing, current_easting, position.heading,
                            current_odometer - start_odometer])
            print(f"Total traveled distance: {current_odometer - start_odometer}")

        elif isinstance(msg, telemetry_pb2.CalibratedImuTel):
            imu, imu_ts = msg.imu, ts
            imu_log.append([count, imu_ts, imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z,
                            imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])

    np.savetxt(os.path.join(args.out, "dvl_odometer.csv"), np.array(dvl_log), delimiter=",",
               header="frame_id,ts,northing,easting,heading,odometer,")

    np.savetxt(os.path.join(args.out, "imu.csv"), np.array(imu_log), delimiter=",",
               header="frame_id,ts,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z")

    pipeline.redraw(realignment, redraw_id)
    plt.imsave(os.path.join(args.out, f"combined.png"), pipeline.combined, cmap="gray")
    fig = plot_slam2d(pipeline.pose_graph.optimizer, "Optimized pose graph")
    fig.write_image(os.path.join(args.out, f"graph.png"))
    fig.write_html(os.path.join(args.out, f"graph.html"))

    # pipeline.optimize(10000, verbose=args.verbose)
    #
    # fig = plot_slam2d(pipeline.pose_graph.optimizer, "Final optimisation")
    # fig.write_image(os.path.join(args.out, f"final_graph.png"))
    # fig.write_html(os.path.join(args.out, f"final_graph.html"))
    # pipeline.redraw(realignment, redraw_id)
    # plt.imsave(os.path.join(args.out, f"final_combined.png"), pipeline.combined, cmap="gray")


if __name__ == '__main__':
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    main()
