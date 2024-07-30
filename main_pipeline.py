#!/usr/bin/env python
import os
import sys
import signal
import time
from enum import Enum
import argparse

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


def get_fake_data(path, start_frame, max_frames):
    Object = lambda **kwargs: type("Object", (), kwargs)
    i = 0
    while True:
        try:
            img = plt.imread(os.path.join(path, f'raw_{i}.png'))
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            yield Object(bearings=np.array([theta for theta in np.linspace(-60, 60, 520)]), number_of_ranges=520,
                         range=20), img, i * 0.1
            i += 1
            if max_frames is not None and i > max_frames:
                break
        except FileNotFoundError as e:
            break
    return


class Algorithm(Enum):
    FOURIER_MELLIN = 1
    PHASE_CORRELATION = 2


def create_fourier_mellin(args, bearings, fov, height, width, range_resolution):

    pipeline = Pipeline(verbose=args.verbose)

    conditioning_id, conditioning = pipeline.add_module(Pipeline('conditioning'))
    pipeline.add_module(MetricsModule(output=args.out))
    filtering_id, filtering = pipeline.add_module(Pipeline('filtering'))
    registration_id, registration = pipeline.add_module(Pipeline('registration'))
    alignment_id, alignment = pipeline.add_module(Pipeline('global_alignment'))
    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=conditioning_id)
    # pipeline.add_module(WarpModule(combine=True), 'b')
    # pipeline.add_module(IdentityModule(), input_stage=filtering_id)
    pipeline.add_module(OdometerModule(output=args.out, range_resolution=range_resolution))

    if not args.disable_resizing:
        conditioning.add_module(ResizeModule(args.resize))
    conditioning.add_module(FanModule2(bearings, output=args.out))
    conditioning.add_module(PaddingModule(4))

    # filtering.add_module(BandpassModule(2, 20))
    # filtering.add_module(BandpassTestingModule(output=out))
    filtering.add_module(BandpassModule(args.bandpass_low, args.bandpass_high))
    filtering.add_module(MaskModule(padding=60, sigma=15))

    start_id, _ = registration.add_module(IdentityModule())
    registration.add_module(FourierModule())
    registration.add_module(LogPolarModule(order=1))
    registration.add_module(PhaseCorrelationModule(10, 'rotation'))
    registration.add_module(IdentityModule(), input_stage=start_id)
    registration.add_module(WarpModule(), apply_to=('b', 'm'))
    registration.add_module(PhaseCorrelationModule(10, 'translation'))

    alignment.add_module(UpdateTformModule(range_resolution=range_resolution))
    if not args.disable_loop_closure:
        alignment.add_module(FindNeighborsModule(registration_id, range_resolution=range_resolution, output=args.out,
                                                 matches_number=args.matches_number,
                                                 error_threshold=args.error_threshold),
                             input_stage=filtering_id)
    # alignment.add_module(IdentityModule(), input_stage=registration_id)

    realignment = Pipeline()
    realignment.add_module(WarpModule(combine=True), 'b')

    return pipeline, filtering_id, realignment


def create_phase_correlation(args, bearings, fov, height, width, range_resolution):
    pipeline = Pipeline(verbose=args.verbose)
    registration_id, registration = pipeline.add_module(Pipeline('registration'))
    if not args.disable_resizing:
        registration.add_module(ResizeModule(args.resize))
    resize_id, _ = registration.add_module(IdentityModule())
    # registration.add_module(PaddingModule(4))
    # registration.add_module(BandpassModule(args.bandpass_low, args.bandpass_high))
    registration.add_module(MaskModule(padding=60, sigma=15))
    # registration.add_module(FourierModule())
    # registration.add_module(LogPolarModule(order=1, radius_factor=1))
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
    # pipeline.add_module(WarpModule(combine=True), 'b')

    realignment = Pipeline()
    realignment.add_module(ResizeModule(args.resize))
    realignment.add_module(FanModule2(bearings, output=args.out))
    realignment.add_module(PaddingModule(4))
    realignment.add_module(WarpModule(combine=True), 'b')

    return pipeline, registration_id, realignment


# Press the green button in the gutter to run the script.
def main():
    global EXIT_FLAG
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the log file")
    parser.add_argument("--out", help="Output directory", default="./out")
    parser.add_argument("--odometry_log_file", help="Read odometry data from a bez file")
    parser.add_argument("--output_frequency", help="Output frequency. How many frames before output is redrawn",
                        type=int, default=25)
    parser.add_argument("--bandpass_low", help="Low cutoff value for bandpass filter", type=int, default=2)
    parser.add_argument("--bandpass_high", help="High cutoff value for bandpass filter", type=int, default=20)
    parser.add_argument("--matches_number", help="Number of matches to consider", type=int, default=10)
    parser.add_argument("--start_frame", help="Start frame", type=int, default=0)
    parser.add_argument("--max_frames", help="Max frames", type=int, default=None)
    parser.add_argument("--verbose", help="Verbose", action="store_true")
    parser.add_argument("--resize", help="Resize the images", type=int, default=50000)
    parser.add_argument("--disable_resizing", help="Disable resizing", action="store_true", default=False)
    parser.add_argument("--disable_loop_closure", help="Disable loop closure", action="store_true", default=False)
    parser.add_argument("--disable_realignment", help="Disable realignment", action="store_true", default=False)
    parser.add_argument("--error-threshold", help="Error threshold for loop closure matches", type=int, default=1)
    parser.add_argument("--fmt", help="Use fourier-mellin transform", action="store_const",
                        const=Algorithm.FOURIER_MELLIN, dest="algorithm")
    parser.add_argument("--pc", help="Use phase correlation", action="store_const", const=Algorithm.PHASE_CORRELATION,

                        dest="algorithm")

    # Parse arguments
    args = parser.parse_args()

    # make sure the output directory exists
    os.makedirs(args.out, exist_ok=True)

    # Open generator
    if args.path.endswith(".bez") or args.path.endswith(".mbez"):
        generator = binlog.read_ping_2(args.path, start_frame=args.start_frame, max_frames=args.max_frames)
    else:
        generator = get_fake_data(args.path, start_frame=args.start_frame, max_frames=args.max_frames)

    # Open odometry generator
    odometry_generator = binlog.read_binlog(args.odometry_log_file,
                                            message=telemetry_pb2.PositionEstimateTel) if args.odometry_log_file else None

    # Read the first ping
    a_ping, a_raw, a_ts = next(generator)

    prev_northing = prev_easting = start_odometer = 0
    if odometry_generator:
        position, pos_ts = next(odometry_generator)
        while a_ts > pos_ts:
            position, pos_ts = next(odometry_generator)

        prev_northing = position.position_estimate.northing
        prev_easting = position.position_estimate.easting
        start_odometer = position.position_estimate.odometer

    bearings = a_ping.bearings
    fov = np.max(bearings) - np.min(bearings)
    height = a_ping.number_of_ranges
    width = int(2 * a_ping.number_of_ranges * np.sin(np.deg2rad(fov / 2)))
    total_size = a_raw.shape[0] * a_raw.shape[1]
    ratio = np.sqrt(args.resize / total_size)
    range_resolution = (a_ping.range / (a_ping.number_of_ranges * ratio))
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

    count = 1
    current_northing = current_easting = current_odometer = 0
    dvl_log = []
    while True:
        if EXIT_FLAG:
            break

        try:
            b_ping, b_raw, b_ts = next(generator)
        except StopIteration as e:
            print("End of file")
            break

        print(f"\nProcessing pings {count - 1} and {count}")
        pipeline.execute(a_raw, b_raw)

        if odometry_generator:
            position, pos_ts = next(odometry_generator)

            while b_ts + 0.5 > pos_ts:
                position, pos_ts = next(odometry_generator)

            current_northing = position.position_estimate.northing
            current_easting = position.position_estimate.easting
            current_odometer = position.position_estimate.odometer

            scaled_northing = (current_northing - prev_northing) / range_resolution
            scaled_easting = (current_easting - prev_easting) / range_resolution
            # dvl_log.append()

            # pipeline.pose_graph.add_landmark_to_last_vertex(scaled_northing, scaled_easting)
            prev_northing = current_northing
            prev_easting = current_easting

            dvl_log.append([current_northing, current_easting, position.position_estimate.heading, current_odometer - start_odometer])
            print(f"Total traveled distance: {current_odometer - start_odometer}")

        if pipeline.verbose:
            fig = plot_slam2d(pipeline.pose_graph.optimizer, "Before optimisation")
            fig.write_image(os.path.join(args.out, f"graph_{count}.png"))
            fig.write_html(os.path.join(args.out, f"graph_{count}.html"))

        if not args.disable_realignment and count % 10 == 0:
            pipeline.optimize(100, verbose=args.verbose)

        if pipeline.verbose:
            fig = plot_slam2d(pipeline.pose_graph.optimizer, "After optimisation")
            fig.write_image(os.path.join(args.out, f"graph_{count}_optimized.png"))
            fig.write_html(os.path.join(args.out, f"graph_{count}_optimized.html"))

        if count % args.output_frequency == 0:
            pipeline.redraw(realignment, redraw_id)
            plt.imsave(os.path.join(args.out, f"combined_{count:08}.png"), pipeline.combined, cmap="gray")

        # if count % (max_frames / 5 if max_frames is not None else 50) == 0:
        # plt.imsave(os.path.join(args.out, f"combined_{count:08}.png"), pipeline.combined, cmap="gray")
        count += 1
        a_ping, a_raw, a_ts = b_ping, b_raw, b_ts

    np.savetxt(os.path.join(args.out, "dvl_odometer.csv"), np.array(dvl_log), delimiter=",", header="northing,easting,heading,odometer")

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
