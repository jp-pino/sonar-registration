#!/usr/bin/env python
import os
import time
import csv

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
    def __init__(self, name=__name__, output='./out', verbose=False, intermediate_output=None):
        super().__init__()
        self.name = name
        self.output = output
        self.verbose = verbose
        if intermediate_output is None:
            intermediate_output = []
        self.intermediate_output = intermediate_output

        self.modules = []
        self.outputs = {}
        self.source_cache = []

        self.timing = []

        self.total_tform = SimilarityTransform()
        self.centering_tform = SimilarityTransform()
        self.combined = None
        self.combined_count = 0

        self.pose_graph = PoseGraph(verbose=verbose)
        self.pose_graph.add_vertex(g2o.SE2(), set_last=True, fixed=True)

        self.executing = False

        self.origin = None

    def __del__(self):
        print(f"Deleting pipeline {self.name}")
        if self.find_root() == self:
            print("Saving timing")
            self.save_file(f"{self.name}_timing.csv")

    def save_file(self, path):
        with open(os.path.join(self.output, path), 'w') as f:
            write = csv.writer(f)

            write.writerow(["module", "time"])
            write.writerows(self.timing)

    def read_cache(self, cache_id):
        a, b, mask = self.source_cache[cache_id]
        return a.copy(), b.copy(), mask.copy()

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
        if name == self.name:
            return self
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
        self.executing = True
        mask = np.ones_like(a)
        tform = SimilarityTransform()
        error = [0, 0, 0]

        a, b, mask, tform, error = self.run(a, b, mask, tform, error)

        self.executing = False

        return a, b, mask, tform, error, self.total_tform

    def run(self, a, b, mask, tform, error):
        start_time = time.time()
        if self.executing:
            self.source_cache.append((a.copy(), b.copy(), mask.copy()))
            print(f"Appending to source cache. Length: {len(self.source_cache)}")
        self.find_root().outputs[self.name] = (a.copy(), b.copy(), mask.copy())
        self.find_root().outputs['source'] = (a.copy(), b.copy(), mask.copy())
        self.find_root().outputs['chain'] = (a.copy(), b.copy(), mask.copy())

        if self.find_root().origin is None:
            width = a.shape[1]
            height = a.shape[0]
            self.find_root().origin = np.array([width // 2, height // 2])
        for module, apply_to, input_stage, output in self.modules:
            start = time.time()

            if input_stage not in self.find_root().outputs.keys():
                raise ValueError(f"Invalid input_stage: {input_stage}")

            a_tmp, b_tmp, mask_tmp = self.find_root().outputs[input_stage]
            if a_tmp is None or b_tmp is None or mask_tmp is None:
                raise ValueError(f"Invalid input_stage: {input_stage}")

            a_tmp = a_tmp.copy() if 'a' in apply_to else None
            b_tmp = b_tmp.copy() if 'b' in apply_to else None
            mask_tmp = mask_tmp.copy() if 'm' in apply_to else mask.copy()

            if self.executing:
                module.executing = True
            a_tmp, b_tmp, mask_tmp, tform, error = module.run(a_tmp, b_tmp, mask_tmp, tform, error)
            if self.executing:
                module.executing = False

            a = a_tmp if 'a' in apply_to else a
            b = b_tmp if 'b' in apply_to else b
            mask = mask_tmp if 'm' in apply_to else mask

            self.find_root().outputs[module.name] = (a.copy(), b.copy(), mask.copy())
            self.find_root().outputs['chain'] = (a.copy(), b.copy(), mask.copy())

            if module.name in self.find_root().intermediate_output:
                output = self.find_root().output

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
            self.find_root().timing.append([module.name, total_time])

        self.find_root().outputs[self.name] = (a.copy(), b.copy(), mask.copy())
        pipeline_time = time.time() - start_time
        print(f"  > Total time for pipeline {self.name}: {time.time() - start_time} seconds")
        self.find_root().timing.append([self.name, pipeline_time])

        return a, b, mask, tform, error

    def redraw(self, pipeline, input_id):
        self.total_tform = self.centering_tform = SimilarityTransform()
        self.combined = None
        self.combined_count = 0
        name = pipeline.name
        pipeline_id, _ = self.add_module(pipeline)
        error = [0, 0, 0]
        for vertex in reversed(self.pose_graph.optimizer.vertices().values()):
            if type(vertex) != g2o.VertexSE2:
                continue

            if vertex.id() == 0:
                continue

            if vertex.id() >= len(self.get_module(input_id).source_cache):
                continue

            x, y, theta = vertex.estimate().to_vector()
            print(f"Vertex {vertex.id()} at {x}, {y}, {np.rad2deg(theta)}")

            a, b, mask = self.get_module(input_id).read_cache(vertex.id() - 1)

            tform = SimilarityTransform(translation=[x, y], rotation=theta)
            self.total_tform = tform

            pipeline.run(a, b, mask, tform, error)

        self.remove_module(pipeline_id)
        pipeline.name = name

    def optimize(self, iterations=10, verbose=False):
        if verbose:
            for vertex in reversed(self.pose_graph.optimizer.vertices().values()):
                if type(vertex) != g2o.VertexSE2:
                    continue
                print(f"Vertex {vertex.id()} at {vertex.estimate().to_vector()}")
            for edge in self.pose_graph.optimizer.edges():
                if type(edge) != g2o.EdgeSE2:
                    continue
                print(f"Edge {edge.id()} at {edge.measurement().to_vector()}")

        start_time = time.time()
        print("Optimizing...")
        self.pose_graph.get_last().set_fixed(True)
        self.pose_graph.optimize(iterations, verbose=verbose)
        self.pose_graph.get_last().set_fixed(False)
        print(f"Optimization took {time.time() - start_time} seconds")

        if verbose:
            for vertex in reversed(self.pose_graph.optimizer.vertices().values()):
                if type(vertex) != g2o.VertexSE2:
                    continue
                print(f"Realigned Vertex {vertex.id()} at {vertex.estimate().to_vector()}")
            for edge in self.pose_graph.optimizer.edges():
                if type(edge) != g2o.EdgeSE2:
                    continue
                print(f"Realigned Edge {edge.id()} at {edge.measurement().to_vector()}")
