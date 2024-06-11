import os

import matplotlib.pyplot as plt
import numpy as np
import ray
from skimage.transform import SimilarityTransform

from pipeline.PipelineModule import PipelineModule
from alignment.PoseGraph import PoseGraph


class FindNeighborsModule(PipelineModule):
    def __init__(self, registration_id, delta_t=40, delta_r=40, error_threshold=0.7, output=None):
        super().__init__()
        self.registration_id = registration_id
        self.delta_t = delta_t
        self.delta_r = delta_r
        self.error_threshold = error_threshold
        self.output = output

    def run(self, a, b, mask, tform, error):
        current = self.find_root().pose_graph.get_last().id()
        matches = self.find_root().pose_graph.find_possible_matches(current, self.delta_t, self.delta_r)
        print(f"    > Found {len(matches)} possible matches: {matches}")

        for match in matches:
            _, b_match, mask_match = self.find_root().get_module(self.registration_id).source_cache[match - 1]
            _, _, _, tform_neighbor, error_neighbor \
                = self.find_root().get_module(self.registration_id).run(b.copy(), b_match.copy(), mask, SimilarityTransform(), 0)

            if error_neighbor > self.error_threshold:
                print(f"    > Skipping match due to high error: {error_neighbor}")
                out = self.output if self.output is not None else "out"
                plt.imsave(os.path.join(out, f"error_{current}_{match}_{error_neighbor}_c.png"), b, cmap="gray")
                plt.imsave(os.path.join(out, f"error_{current}_{match}_{error_neighbor}_m.png"), b_match, cmap="gray")
                continue

            print(f"    > Original error: {error_neighbor}")
            error_neighbor = ((1 - error_neighbor) * 0.005)
            print(f"    > "
                  f"Match ({current} - {match}) "
                  f"Translation: {tform_neighbor.translation}, "
                  f"Rotation: {tform_neighbor.rotation}, "
                  f"Scale: {tform_neighbor.scale}, "
                  f"Error: {error_neighbor}")
            if self.output is not None:
                plt.imsave(os.path.join(self.output, f"match_{current}_{match}_c.png"), b, cmap="gray")
                plt.imsave(os.path.join(self.output, f"match_{current}_{match}_m.png"), b_match, cmap="gray")

            tform_neighbor = tform_neighbor.inverse
            self.find_root().pose_graph.add_loop_closure_edge(current,
                                                              match,
                                                              tform_neighbor.translation[0],
                                                              tform_neighbor.translation[1],
                                                              tform_neighbor.rotation,
                                                              error_neighbor * np.eye(3))
        return a, b, mask, tform, error


class UpdateTformModule(PipelineModule):
    def run(self, a, b, mask, tform, error):
        print(f"    > Original error: {error}")
        error = ((1 - error) * 0.5 + 0.5)
        print(f"    > "
              f"Translation: {tform.translation}, "
              f"Rotation: {tform.rotation}, "
              f"Scale: {tform.scale}, "
              f"Error: {error}")
        self.find_root().pose_graph.add_odometry(
            tform.translation[0],
            tform.translation[1],
            tform.rotation,
            error * np.eye(3))
        self.find_root().total_tform += tform

        return a, b, mask, tform, error
