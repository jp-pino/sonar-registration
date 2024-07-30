import os
import sys
import signal

from matplotlib import pyplot as plt

from pipeline.Pipeline import Pipeline
from registration.Modules import *
from alignment.Modules import *
from alignment.plot_slam2d import plot_slam2d

from utils import binlog


def signal_handler(sig, frame):
    plt.imsave(os.path.join(out, f"combined_error.png"), pipeline.combined, cmap="gray")
    fig = plot_slam2d(pipeline.pose_graph.optimizer, "Error")
    fig.write_image(os.path.join(out, f"error_graph.png"))
    sys.exit(0)


def get_fake_data(path, start_frame, max_frames):
    for i in range(start_frame, max_frames):
        img = plt.imread(os.path.join(path, f'raw_{i}.png'))
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        yield i, img, 1, 120, 1, i * 0.1
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Take path from command line
    path = sys.argv[1] if len(sys.argv) > 1 else "./logs/log-multibeam.bez"
    out = sys.argv[2] if len(sys.argv) > 2 else "./out"
    start_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end_frame = int(sys.argv[4]) if len(sys.argv) > 4 else None
    signal.signal(signal.SIGINT, signal_handler)

    # make sure the output directory exists
    os.makedirs(out, exist_ok=True)

    # Open generator
    generator = get_fake_data(path, start_frame=start_frame, max_frames=end_frame)

    # Read the first ping
    a_id, a_raw, _, a_aperture, range_resolution, a_ts = next(generator)
    print(f'>>>>>> Shape: {a_raw.shape}')

    pipeline = Pipeline(verbose=True)
    registration_id, registration = pipeline.add_module(Pipeline('registration'))
    resize_id, _ = registration.add_module(IdentityModule())
    registration.add_module(PaddingModule(4))
    registration.add_module(MaskModule(padding=50, sigma=15))
    registration.add_module(LogPolarModule(order=1))
    registration.add_module(PhaseCorrelationModule(10, 'rotation', log_polar=True))
    registration.add_module(FanModule(a_aperture), input_stage=resize_id)
    padding_id, _ = registration.add_module(PaddingModule(4))
    registration.add_module(MaskModule())
    registration.add_module(WarpModule(), apply_to=('b', 'm'))
    registration.add_module(PhaseCorrelationModule(10, 'translation'))

    pipeline.add_module(UpdateTformModule())
    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=pipeline.name)
    # pipeline.add_module(FindNeighborsModule(registration_id, delta_theta=40, delta_radius=40, error_threshold=1), input_stage=pipeline.name)
    pipeline.add_module(IdentityModule(), input_stage=padding_id)
    pipeline.add_module(WarpModule(combine=True))

    count = 0
    while True:
        try:
            b_id, b_raw, _, b_aperture, range_resolution, b_ts = next(generator)
        except StopIteration as e:
            print("End of file")
            break

        print(f"Processing pings {a_id} and {b_id}")
        a, b, mask, tform, error, total_tform = pipeline.execute(a_raw, b_raw)
        if count % (end_frame / 5 if end_frame is not None else 50) == 0:
            plt.imsave(os.path.join(out, f"combined_{b_id:08}.png"), pipeline.combined, cmap="gray")

        print(f"A size: {a_raw.shape}, B size: {b_raw.shape}")
        print(f"Combined size: {pipeline.combined.shape}")
        count += 1
        a_id, a_raw, _, a_aperture, a_ts = b_id, b_raw, _, b_aperture, b_ts

    plt.imsave(os.path.join(out, f"combined.png"), pipeline.combined, cmap="gray")
    fig = plot_slam2d(pipeline.pose_graph.optimizer, "Before optimisation")
    fig.write_image(os.path.join(out, f"b_graph.png"))
    fig.write_html(os.path.join(out, f"b_graph.html"))
    #
    # realignment = Pipeline()
    # realignment.add_module(IdentityModule())
    # realignment.add_module(FanModule(a_aperture))
    # realignment.add_module(PaddingModule(4))
    # realignment.add_module(WarpModule(combine=True), 'b')
    #
    # pipeline.redraw(realignment, pipeline.name)
    # plt.imsave(os.path.join(out, f"realigned_1.png"), pipeline.combined, cmap="gray")
    #
    # pipeline.optimize(10000, verbose=True)
    #
    # fig = plot_slam2d(pipeline.pose_graph.optimizer, "After optimisation")
    # fig.write_image(os.path.join(out, f"a_graph.png"))
    # fig.write_html(os.path.join(out, f"a_graph.html"))
    #
    # pipeline.redraw(realignment, pipeline.name)
    # plt.imsave(os.path.join(out, f"realigned_2.png"), pipeline.combined, cmap="gray")