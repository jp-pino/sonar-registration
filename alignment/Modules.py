import numpy as np
import ray
from skimage.transform import SimilarityTransform

from pipeline.PipelineModule import PipelineModule
from alignment.PoseGraph import PoseGraph


class FindNeighborsModule(PipelineModule):
    def __init__(self, registration_id):
        super().__init__()
        self.registration_id = registration_id

    def run(self, a, b, mask, tform, error):
        current = self.find_root().pose_graph.get_last().id()
        matches = self.find_root().pose_graph.find_possible_matches(current, 70, 5)
        print(f"    > Found {len(matches)} possible matches: {matches}")

        for match in matches:
            _, b2, mask2 = self.find_root().get_module(self.registration_id).source_cache[match]
            _, _, _, tform_neighbor, error \
                = self.find_root().get_module(self.registration_id).run(b, b2, mask, SimilarityTransform(), 0)

            print(f"    > Match ({current} - {match}) Translation: {tform_neighbor.translation}, Rotation: {tform_neighbor.rotation}, Scale: {tform_neighbor.scale}")
            if error > 0.3:
                continue

            self.find_root().pose_graph.add_loop_closure_edge(current,
                                                              match,
                                                              tform_neighbor.translation[0],
                                                              tform_neighbor.translation[1],
                                                              tform_neighbor.rotation,
                                                              (1 - error) * np.eye(3))
        return a, b, mask, tform, error


class UpdateTformModule(PipelineModule):
    def run(self, a, b, mask, tform, error):
        print(f"    > Translation: {tform.translation}, Rotation: {tform.rotation}, Scale: {tform.scale}")
        self.find_root().pose_graph.add_odometry(tform.translation[0], tform.translation[1], tform.rotation, (1 - error) * np.eye(3))
        self.find_root().total_tform += tform

        return a, b, mask, tform, error
