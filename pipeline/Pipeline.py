import os
import time

import g2o
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from alignment.PoseGraph import PoseGraph

from pipeline.PipelineModule import PipelineModule
from skimage.transform import (
    SimilarityTransform,
)


class Pipeline(PipelineModule):
    def __init__(self, name=__name__, verbose=False):
        super().__init__()
        self.name = name

        self.modules = []
        self.outputs = {}
        self.source_cache = []

        self.total_tform = SimilarityTransform()
        self.centering_tform = SimilarityTransform()
        self.combined = None
        self.combined_count = 0

        self.verbose = verbose
        self.pose_graph = PoseGraph(verbose=verbose)
        self.pose_graph.add_fixed_pose(g2o.SE2())

    def add_module(self, module: PipelineModule, apply_to=('a', 'b', 'm'), input_stage='chain', output=None):
        name = f"{self.name}_{module.name}__{len(self.modules)}"

        print(f"Adding module {name} with apply_to={apply_to} and input_stage={input_stage}")
        for a in apply_to:
            if a not in ['a', 'b', 'm']:
                raise ValueError(f"Invalid value for apply_to: {a}")

        if (input_stage not in ['chain', 'source']
                and self.find_root().get_module(input_stage) is None):
            for module, _, _, _ in self.modules:
                print(f"  > {module.name}")
            raise ValueError(f"Invalid value for input_stage: {input_stage}")

        module.pipeline = self
        module.name = name
        self.modules.append((module, apply_to, input_stage, output))
        return name, module

    def remove_module(self, name):
        for i, (module, _, _, _) in enumerate(self.modules):
            if module.name == name:
                print(f"Removing module {name}")
                self.modules.pop(i)
                return

    def get_module(self, name):
        for module, _, _, _ in self.modules:
            # Check if the module is the one we are looking for
            if module.name == name:
                return module

            # Recursively search for the module
            res = module.get_module(name)
            if res is not None:
                return res

        return None

    def get_module_by_type(self, class_name):
        modules = []
        for module, _, _, _ in self.modules:
            if module.__class__.__name__ == class_name:
                modules.append(module)

            # Recursively search for the module
            modules += module.get_module_by_type(class_name)

        return modules

    def get_global_tform(self):
        return self.centering_tform + self.total_tform.inverse

    def execute(self, a: np.ndarray, b: np.ndarray):
        mask = np.ones_like(a)
        tform = SimilarityTransform()
        error = None

        a, b, mask, tform, error = self.run(a, b, mask, tform, error)

        return a, b, mask, tform, error, self.total_tform

    def run(self, a, b, mask, tform, error):
        start_time = time.time()
        self.source_cache.append((a.copy(), b.copy(), mask.copy()))
        self.find_root().outputs['source'] \
            = self.find_root().outputs['chain'] \
            = (self.source_cache[-1][0], self.source_cache[-1][1], mask)
        for module, apply_to, input_stage, output in self.modules:
            start = time.time()

            if input_stage not in self.find_root().outputs.keys():
                raise ValueError(f"Invalid input_stage: {input_stage}")

            a_tmp, b_tmp, mask_tmp = self.find_root().outputs[input_stage]

            a_tmp = a_tmp.copy() if 'a' in apply_to else None
            b_tmp = b_tmp.copy() if 'b' in apply_to else None
            mask_tmp = mask_tmp.copy() if 'm' in apply_to else mask.copy()

            a_tmp, b_tmp, mask_tmp, tform, error = module.run(a_tmp, b_tmp, mask_tmp, tform, error)

            a = a_tmp if 'a' in apply_to else a
            b = b_tmp if 'b' in apply_to else b
            mask = mask_tmp if 'm' in apply_to else mask

            self.find_root().outputs[self.name] \
                = self.find_root().outputs[module.name] \
                = self.find_root().outputs['chain'] = (a.copy(), b.copy(), mask.copy())

            if output is not None:
                if 'a' in apply_to:
                    plt.imsave(os.path.join(output, f"{start_time}_{start}_a_{module.name}.png"), a,
                               cmap="gray")
                if 'b' in apply_to:
                    plt.imsave(os.path.join(output, f"{start_time}_{start}_b_{module.name}.png"), b,
                               cmap="gray")
                if 'm' in apply_to:
                    plt.imsave(os.path.join(output, f"{start_time}_{start}_mask_{module.name}.png"), mask,
                               cmap="gray")

            total_time = time.time() - start
            color = Fore.RED if total_time > 0.2 else Fore.GREEN
            print(f"{color}  > Module {module.name} took {total_time} seconds{Style.RESET_ALL}")

        print(f"  > Total time for pipeline {self.name}: {time.time() - start_time} seconds")

        return a, b, mask, tform, error

    def redraw(self, pipeline, input_id):
        self.total_tform = self.centering_tform = SimilarityTransform()
        self.combined = None
        self.combined_count = 0
        pipeline_id, _ = self.add_module(pipeline)
        error = None
        for vertex in reversed(self.pose_graph.optimizer.vertices().values()):
            if vertex.id() == 0:
                continue

            if vertex.id() >= len(self.get_module(input_id).source_cache):
                continue

            x, y, theta = vertex.estimate().to_vector()
            print(f"Vertex {vertex.id()} at {x}, {y}, {np.rad2deg(theta)}")

            a, b, mask = self.get_module(input_id).source_cache[vertex.id() - 1]

            # center = np.array(a.shape) // 2
            # tform = SimilarityTransform()
            # tform += SimilarityTransform(translation=-center)
            # tform += SimilarityTransform(scale=1, rotation=theta)
            # tform += SimilarityTransform(translation=center)
            # tform += SimilarityTransform(translation=[x, y])
            tform = SimilarityTransform(translation=[x, y], rotation=theta)
            self.total_tform = tform

            pipeline.run(a, b, mask, tform, error)

        self.remove_module(pipeline_id)

    def optimize(self, iterations=10, verbose=False):
        for vertex in reversed(self.pose_graph.optimizer.vertices().values()):
            print(f"Vertex {vertex.id()} at {vertex.estimate().to_vector()}")
        for edge in self.pose_graph.optimizer.edges():
            print(f"Edge {edge.id()} at {edge.measurement().to_vector()}")
        self.total_tform = self.centering_tform = SimilarityTransform()
        self.pose_graph.optimize(iterations, verbose=verbose)
        for vertex in reversed(self.pose_graph.optimizer.vertices().values()):
            print(f"Realigned Vertex {vertex.id()} at {vertex.estimate().to_vector()}")
        for edge in self.pose_graph.optimizer.edges():
            print(f"Realigned Edge {edge.id()} at {edge.measurement().to_vector()}")



