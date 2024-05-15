import os
import sys

from matplotlib import pyplot as plt

from registration.Pipeline import Pipeline
from registration.Modules import *

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


    conditioning = Pipeline()
    conditioning.add_module(ResizeModule(90000))
    conditioning.add_module(FanModule(a_aperture))
    conditioning.add_module(PaddingModule(4))

    filtering = Pipeline()
    filtering.add_module(BandpassModule(5, 20))
    filtering.add_module(MaskModule(padding=50))

    rotation = Pipeline()
    rotation.add_module(FourierModule())
    rotation.add_module(LogPolarModule(order=1))
    rotation.add_module(PhaseCorrelationModule(10, 'rotation'))

    translation = Pipeline()
    translation.add_module(WarpModule(), apply_to=('b', 'm'))
    translation.add_module(PhaseCorrelationModule(10, 'translation'))

    pipeline = Pipeline()
    conditioning_id = pipeline.add_module(conditioning)
    filtering_id = pipeline.add_module(filtering)
    pipeline.add_module(rotation)
    pipeline.add_module(IdentityModule(), input_stage=filtering_id)
    pipeline.add_module(translation)
    pipeline.add_module(IdentityModule(), ('a', 'b'), input_stage=conditioning_id)
    pipeline.add_module(UpdateTformModule())
    pipeline.add_module(WarpModule(combine=True), 'b')

    count = 0
    while True:
        b_id, b_raw, _, b_aperture, b_ts = next(generator)

        print(f"Processing pings {a_id} and {b_id}")
        a, b, mask, tform, total_tform = pipeline.execute(a_raw, b_raw)
        if count % 10 == 0:
            plt.imsave(os.path.join(out, f"combined_{b_id:08}.png"), pipeline.combined, cmap="gray")
            # plt.imsave(os.path.join(out, f"mask_{b_id:08}.png"), mask, cmap="gray")
            # plt.imsave(os.path.join(out, f"a_{b_id:08}.png"), a, cmap="gray")
            # plt.imsave(os.path.join(out, f"b_{b_id:08}.png"), b, cmap="gray")

        print(f"A size: {a_raw.shape}, B size: {b_raw.shape}")
        print(f"Combined size: {pipeline.combined.shape}")
        count += 1
        a_id, a_raw, _, a_aperture, a_ts = b_id, b_raw, _, b_aperture, b_ts

