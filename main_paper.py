import os
import sys

from matplotlib import pyplot as plt

from pipeline.Pipeline import Pipeline
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


    # pipeline = Pipeline()
    # pipeline.add_module(ResizeModule(100000))
    # pipeline.add_module(FanModule(a_aperture))
    # pipeline.add_module(PaddingModule(4))
    # pipeline.add_module(BandpassModule(5, 20))
    # pipeline.add_module(MaskModule(padding=50))
    # pipeline.add_module(FourierModule())
    # pipeline.add_module(LogPolarModule(order=1))
    # pipeline.add_module(PhaseCorrelationModule(10, 'rotation'))
    # pipeline.add_module(WarpModule(), apply_to=('b', 'm'), input_stage=MaskModule.__name__)
    # pipeline.add_module(IdentityModule(), ('a'), input_stage=MaskModule.__name__)
    # pipeline.add_module(PhaseCorrelationModule(10, 'translation'))
    # pipeline.add_module(IdentityModule(), ('a'), input_stage=PaddingModule.__name__)
    # pipeline.add_module(UpdateTformModule())
    # pipeline.add_module(WarpModule(combine=True), ('b'), input_stage=PaddingModule.__name__)

    pipeline = Pipeline()
    pipeline.add_module(ResizeModule(100000))
    pipeline.add_module(PaddingModule(4))
    pipeline.add_module(MaskModule())
    pipeline.add_module(LogPolarModule(order=3))
    pipeline.add_module(PhaseCorrelationModule(10, 'rotation', invert=True))
    pipeline.add_module(FanModule(a_aperture), input_stage=ResizeModule.__name__)
    pipeline.add_module(PaddingModule(4))
    pipeline.add_module(MaskModule())
    pipeline.add_module(WarpModule(), apply_to=('b', 'm'))
    pipeline.add_module(PhaseCorrelationModule(10, 'translation', normalization='phase'))
    pipeline.add_module(UpdateTformModule())
    pipeline.add_module(WarpModule(combine=True), ('b', 'm'), input_stage=PaddingModule.__name__)

    count = 0
    while True:
        b_id, b_raw, _, b_aperture, b_ts = next(generator)

        print(f"Processing pings {a_id} and {b_id}")
        a, b, mask, tform, total_tform = pipeline.run(a_raw, b_raw)
        if count % 10 == 0:
            plt.imsave(os.path.join(out, f"combined_{b_id:08}.png"), pipeline.combined, cmap="gray")

        print(f"A size: {a_raw.shape}, B size: {b_raw.shape}")
        print(f"Combined size: {pipeline.combined.shape}")
        count += 1
        a_id, a_raw, _, a_aperture, a_ts = b_id, b_raw, _, b_aperture, b_ts

