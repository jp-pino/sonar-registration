import os
import sys

from matplotlib import pyplot as plt

from pipeline.Pipeline import Pipeline
from registration.Modules import *
from alignment.Modules import *
from alignment.plot_slam2d import plot_slam2d

from utils import binlog

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
    a_id, a_raw, _, a_aperture, a_ts = next(generator)

    pipeline = Pipeline()
    conditioning_id, conditioning = pipeline.add_module(Pipeline('conditioning'))
    filtering_id, filtering = pipeline.add_module(Pipeline('filtering'))
    registration_id, registration = pipeline.add_module(Pipeline('registration'))

    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=conditioning_id)
    pipeline.add_module(WarpModule(combine=True), 'b')

    conditioning.add_module(ResizeModule(90000))
    conditioning.add_module(FanModule(a_aperture))
    conditioning.add_module(PaddingModule(4))

    filtering.add_module(BandpassModule(5, 20))
    filtering.add_module(MaskModule(padding=50))

    registration.add_module(FourierModule())
    registration.add_module(LogPolarModule(order=1))
    registration.add_module(PhaseCorrelationModule(10, 'rotation'))
    registration.add_module(IdentityModule(), input_stage=filtering_id)
    registration.add_module(WarpModule(), apply_to=('b', 'm'))
    registration.add_module(PhaseCorrelationModule(10, 'translation'))
    registration.add_module(UpdateTformModule())

    count = 0
    while True:
        b_id, b_raw, _, b_aperture, b_ts = next(generator)

        print(f"Processing pings {a_id} and {b_id}")
        a, b, mask, tform, total_tform = pipeline.execute(a_raw, b_raw)
        if count % 30 == 0:
            plt.imsave(os.path.join(out, f"combined_{b_id:08}.png"), pipeline.combined, cmap="gray")
            # plt.imsave(os.path.join(out, f"mask_{b_id:08}.png"), mask, cmap="gray")
            # plt.imsave(os.path.join(out, f"a_{b_id:08}.png"), a, cmap="gray")
            # plt.imsave(os.path.join(out, f"b_{b_id:08}.png"), b, cmap="gray")
            fig = plot_slam2d(pipeline.pose_graph.optimizer, "Before optimisation")
            fig.write_image(os.path.join(out, f"graph_{b_id:08}.png"))

        print(f"A size: {a_raw.shape}, B size: {b_raw.shape}")
        print(f"Combined size: {pipeline.combined.shape}")
        count += 1
        a_id, a_raw, _, a_aperture, a_ts = b_id, b_raw, _, b_aperture, b_ts
