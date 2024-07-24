import os
import sys
import signal

import numpy as np
from matplotlib import pyplot as plt

from pipeline.Pipeline import Pipeline
from registration.Modules import *
from alignment.Modules import *
from alignment.plot_slam2d import plot_slam2d

from utils import binlog


# def signal_handler(sig, frame):
#     plt.imsave(os.path.join(out, f"combined_error.png"), pipeline.combined, cmap="gray")
#     fig = plot_slam2d(pipeline.pose_graph.optimizer, "Error")
#     fig.write_image(os.path.join(out, f"error_graph.png"))
#     sys.exit(0)

def get_fake_data(path, start_frame, max_frames):
    Object = lambda **kwargs: type("Object", (), kwargs)
    i = start_frame
    while True:
        try:
            img = plt.imread(os.path.join(path, f'raw_{i}.png'))
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            yield Object(bearings=np.array([theta for theta in np.linspace(-60, 60, 520)]), number_of_ranges=601, range=20), img, i * 0.1
            i += 1
        except FileNotFoundError as e:
            break
    return


# Press the green button in the gutter to run the script.
def main():
    # Take path from command line
    path = sys.argv[1] if len(sys.argv) > 1 else "./logs/log-multibeam.bez"
    out = sys.argv[2] if len(sys.argv) > 2 else "./out"
    start_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
    # signal.signal(signal.SIGINT, signal_handler)

    # make sure the output directory exists
    os.makedirs(out, exist_ok=True)

    # Open generator
    generator = None
    if path.endswith(".bez") or path.endswith(".mbez"):
        generator = binlog.read_ping_2(path, start_frame=start_frame, max_frames=max_frames)
    else:
        generator = get_fake_data(path, start_frame=start_frame, max_frames=max_frames)

    # Read the first ping
    a_ping, a_raw, a_ts = next(generator)

    FOV = np.max(a_ping.bearings) - np.min(a_ping.bearings)
    HEIGHT = a_ping.number_of_ranges
    WIDTH = int(2 * a_ping.number_of_ranges * np.sin(np.deg2rad(FOV / 2)))
    RANGE_RESOLUTION = a_ping.range / a_ping.number_of_ranges


    pipeline = Pipeline(verbose=True)

    conditioning_id, conditioning = pipeline.add_module(Pipeline('conditioning'))
    pipeline.add_module(MetricsModule(output=out))
    filtering_id, filtering = pipeline.add_module(Pipeline('filtering'))
    registration_id, registration = pipeline.add_module(Pipeline('registration'))
    alignment_id, alignment = pipeline.add_module(Pipeline('global_alignment'))
    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=conditioning_id)
    pipeline.add_module(WarpModule(combine=True), 'b')
    pipeline.add_module(IdentityModule(), input_stage=filtering_id)
    pipeline.add_module(OdometerModule(output=out, range_resolution=RANGE_RESOLUTION))


    conditioning.add_module(ResizeModule(85000))
    conditioning.add_module(FanModule2(a_ping.bearings))
    conditioning.add_module(PaddingModule(4))

    # filtering.add_module(BandpassModule(2, 20))
    # filtering.add_module(BandpassTestingModule(output=out))
    filtering.add_module(BandpassModule(2, 20))
    filtering.add_module(MaskModule(padding=50, sigma=15))

    start_id, _ = registration.add_module(IdentityModule())
    registration.add_module(FourierModule())
    registration.add_module(LogPolarModule(order=1))
    registration.add_module(PhaseCorrelationModule(10, 'rotation'))
    registration.add_module(IdentityModule(), input_stage=start_id)
    registration.add_module(WarpModule(), apply_to=('b', 'm'))
    registration.add_module(PhaseCorrelationModule(10, 'translation'))

    alignment.add_module(UpdateTformModule(range_resolution=RANGE_RESOLUTION))
    alignment.add_module(FindNeighborsModule(registration_id, range_resolution=RANGE_RESOLUTION, output=out), input_stage=filtering_id)
    # alignment.add_module(IdentityModule(), input_stage=registration_id)

    realignment = Pipeline()
    realignment.add_module(WarpModule(combine=True), 'b')

    count = 1
    while True:
        try:
            b_ping, b_raw, b_ts = next(generator)
        except StopIteration as e:
            print("End of file")
            break

        print(f"Processing pings {count - 1} and {count}")
        pipeline.execute(a_raw, b_raw)
        if count % 10 == 0:
            fig = plot_slam2d(pipeline.pose_graph.optimizer, "Before optimisation")
            fig.write_image(os.path.join(out, f"graph_{count}.png"))
            fig.write_html(os.path.join(out, f"graph_{count}.html"))
            pipeline.pose_graph.get_last().set_fixed(True)
            pipeline.optimize(50, verbose=False)
            pipeline.pose_graph.get_last().set_fixed(False)
            pipeline.redraw(realignment, filtering_id)
            fig = plot_slam2d(pipeline.pose_graph.optimizer, "After optimisation")
            fig.write_image(os.path.join(out, f"graph_{count}_optimized.png"))
            fig.write_html(os.path.join(out, f"graph_{count}_optimized.html"))
            # break

        # if count % 13 == 0:
        #     pipeline.pose_graph.get_last().set_fixed(True)


        # if count % (max_frames / 5 if max_frames is not None else 50) == 0:
        plt.imsave(os.path.join(out, f"combined_{count:08}.png"), pipeline.combined, cmap="gray")
        count += 1
        a_ping, a_raw, a_ts = b_ping, b_raw, b_ts

    plt.imsave(os.path.join(out, f"combined.png"), pipeline.combined, cmap="gray")
    fig = plot_slam2d(pipeline.pose_graph.optimizer, "Before optimisation")
    fig.write_image(os.path.join(out, f"b_graph.png"))
    fig.write_html(os.path.join(out, f"b_graph.html"))



    # pipeline.redraw(realignment, filtering_id)
    # plt.imsave(os.path.join(out, f"realigned_1.png"), pipeline.combined, cmap="gray")
    #
    # pipeline.optimize(10000, verbose=True)
    #
    # fig = plot_slam2d(pipeline.pose_graph.optimizer, "After optimisation")
    # fig.write_image(os.path.join(out, f"a_graph.png"))
    # fig.write_html(os.path.join(out, f"a_graph.html"))
    #
    # pipeline.redraw(realignment, filtering_id)
    # plt.imsave(os.path.join(out, f"realigned_2.png"), pipeline.combined, cmap="gray")

if __name__ == '__main__':
    main()