import numpy as np
import ray

from pipeline.PipelineModule import PipelineModule
from alignment.PoseGraph import PoseGraph


class FindNeighborsModule(PipelineModule):
    def run(self, a, b, mask, tform):
        origin = self.find_root().total_tform(np.array([[0, 0]]))
        distance_error = 150
        for i, (t, o) in zip(range(len(self.find_root().tforms)), self.find_root().tforms):
            distance = np.linalg.norm(origin - o)
            if distance < distance_error:
                print(f"    > Found neighbor tform with distance {distance}")
        # self.find_root().tforms.append((i, j, tform, origin))
        return a, b, mask, tform

class UpdateTformModule(PipelineModule):
    def run(self, a, b, mask, tform):
        print(f"    > Translation: {tform.translation}, Rotation: {tform.rotation}, Scale: {tform.scale}")
        self.find_root().pose_graph.add_odometry(tform.translation[0], tform.translation[1], tform.rotation, np.eye(3))
        self.find_root().total_tform += tform

        return a, b, mask, tform
