import os
import time

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style

from registration.PipelineModule import PipelineModule
from skimage.transform import (
    SimilarityTransform,
)


class Pipeline:
    def __init__(self):
        self.modules = []
        self.outputs = {}
        self.source_imgs = []

        self.total_tform = SimilarityTransform()
        self.centering_tform = SimilarityTransform()
        self.combined = None
        self.combined_count = 0
        self.padding_top = 0
        self.padding_bottom = 0
        self.padding_left = 0
        self.padding_right = 0

    def add_module(self, module: PipelineModule, apply_to=('a', 'b', 'm'), input_stage='chain', output=None):
        print(f"Adding module {module.__class__.__name__} with apply_to={apply_to} and input_stage={input_stage}")
        for a in apply_to:
            if a not in ['a', 'b', 'm']:
                raise ValueError(f"Invalid value for apply_to: {a}")
        if (input_stage not in ['chain', 'source']
                and input_stage not in [m.__class__.__name__ for m, _, _, _ in self.modules]):
            raise ValueError(f"Invalid value for input_stage: {input_stage}")
        module.set_pipeline(self)
        self.modules.append((module, apply_to, input_stage, output))

    def get_modules(self, class_name):
        modules = []
        for module, _, _, _ in self.modules:
            if module.__class__.__name__ == class_name:
                modules.append(module)
        return modules

    def get_global_tform(self):
        return self.centering_tform + self.total_tform

    def run(self, a: np.ndarray, b: np.ndarray):
        start_time = time.time()
        source_imgs = [a.copy(), b.copy()]
        mask = np.ones_like(a)
        tform = SimilarityTransform()
        self.outputs['source'] = self.outputs['chain'] = (source_imgs[0], source_imgs[1], mask)
        for module, apply_to, input_stage, output in self.modules:
            start = time.time()

            a_tmp, b_tmp, mask_tmp = self.outputs[input_stage]

            a_tmp = a_tmp.copy() if 'a' in apply_to else None
            b_tmp = b_tmp.copy() if 'b' in apply_to else None
            mask_tmp = mask_tmp.copy() if 'm' in apply_to else mask.copy()

            a_tmp, b_tmp, mask_tmp, tform = module.run(a_tmp, b_tmp, mask_tmp, tform)

            a = a_tmp if 'a' in apply_to else a
            b = b_tmp if 'b' in apply_to else b
            mask = mask_tmp if 'm' in apply_to else mask

            self.outputs[module.__class__.__name__] = self.outputs['chain'] = (a.copy(), b.copy(), mask.copy())

            if output is not None:
                if 'a' in apply_to:
                    plt.imsave(os.path.join(output, f"{start_time}_{start}_a_{module.__class__.__name__}.png"), a,
                               cmap="gray")
                if 'b' in apply_to:
                    plt.imsave(os.path.join(output, f"{start_time}_{start}_b_{module.__class__.__name__}.png"), b,
                               cmap="gray")
                if 'm' in apply_to:
                    plt.imsave(os.path.join(output, f"{start_time}_{start}_mask_{module.__class__.__name__}.png"), mask,
                               cmap="gray")

            total_time = time.time() - start
            color = Fore.RED if total_time > 0.2 else Fore.GREEN
            print(f"{color}  > Module {module.__class__.__name__} took {total_time} seconds{Style.RESET_ALL}")

        print(f"  > Total time for pipeline: {time.time() - start_time} seconds")

        return a, b, mask, tform, self.total_tform
