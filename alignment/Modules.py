import os

import g2o
import matplotlib.pyplot as plt
import numpy as np
import ray
from skimage.transform import SimilarityTransform

from pipeline.PipelineModule import PipelineModule
from alignment.PoseGraph import PoseGraph


class FindNeighborsModule(PipelineModule):
    def __init__(self, registration_id, delta_theta=40, delta_radius=40, error_threshold=1, range_resolution=1,
                 output=None):
        super().__init__()
        self.registration_id = registration_id
        self.delta_theta = delta_theta
        self.delta_radius = delta_radius
        self.error_threshold = error_threshold
        self.output = output
        self.range_resolution = range_resolution
        self.log = []

    def __del__(self):
        print("Saving log")
        if self.output is not None:
            print(f"    > Saving to {self.output}")
            np.savetxt(os.path.join(self.output, f"matches.csv"), np.array(self.log), delimiter=",",
                       header="current,match,Δx,Δy,Δθ,error_x,error_y,error_θ")
            print("    > Saved")

    def run(self, a, b, mask, tform, error):
        current = self.find_root().pose_graph.get_last().id()
        print(f"    > Finding neighbors for {current}")
        matches = self.find_root().pose_graph.find_possible_matches(current, self.delta_theta, self.delta_radius)
        print(f"    > Found {len(matches)} possible matches: {matches}")

        # Select N random matches
        select_n = 10
        size = len(matches) if len(matches) < select_n else select_n
        matches = np.random.choice(matches, size, replace=False).tolist()

        # Select previous 5 frames if available
        for i in [current - i for i in range(1, 6 if current > 5 else current)]:
            if i not in matches:
                matches.append(i)

        # Remove id 0 if present
        if 0 in matches:
            matches.remove(0)

        # Select previous 5 frames if available
        # matches = [current - i for i in range(1, 6 if current > 5 else current)]


        for match in matches:
            _, b_match, mask_match = self.find_root().get_module(self.registration_id).source_cache[match - 1]
            _, _, _, tform_neighbor, error_neighbor \
                = self.find_root().get_module(self.registration_id).run(b.copy(),
                                                                        b_match.copy(),
                                                                        mask,
                                                                        SimilarityTransform(),
                                                                        [0, 0, 0])

            if (error_neighbor[0] > self.error_threshold
                    or error_neighbor[1] > self.error_threshold
                    or error_neighbor[2] > self.error_threshold):
                print(f"    > Skipping match due to high error: {error_neighbor}")
                # if self.output is not None:
                #     out = self.output
                #     plt.imsave(os.path.join(out, f"error_{current}_{match}_{error_neighbor}_c.png"), b, cmap="gray")
                #     plt.imsave(os.path.join(out, f"error_{current}_{match}_{error_neighbor}_m.png"), b_match,
                #                cmap="gray")
                continue

            print(f"    > Original error: {error_neighbor}")
            # error_neighbor = np.array(error_neighbor) * 0.25 + 0.75
            # e = np.diag(np.power(error_neighbor, 2))
            # e = np.linalg.inv(np.diag(np.power(error_neighbor, 2)))
            # e = np.linalg.inv((error_neighbor ** 2) * np.eye(3))
            # e = np.diag(np.array(error_neighbor) * 100)
            error_neighbor = 1 - np.array(error_neighbor)
            e = np.diag(np.power(error_neighbor, 2)) / 1000
            print(f"    > "
                  f"Match ({current} - {match}) "
                  f"Translation: {tform_neighbor.translation}, "
                  f"Rotation: {tform_neighbor.rotation}, "
                  f"Scale: {tform_neighbor.scale}, "
                  f"Error: {error_neighbor}")

            self.log.append(
                [current, match, tform_neighbor.translation[0], tform_neighbor.translation[1], tform_neighbor.rotation,
                 error_neighbor[0], error_neighbor[1], error_neighbor[2]])

            # if self.output is not None:
            #     plt.imsave(os.path.join(self.output, f"match_{current}_{match}_c.png"), b.copy(), cmap="gray")
            #     plt.imsave(os.path.join(self.output, f"match_{current}_{match}_m.png"), b_match.copy(), cmap="gray")

            tform_aux = SimilarityTransform(translation=-self.find_root().origin)
            tform_aux += tform_neighbor
            tform_aux += SimilarityTransform(translation=self.find_root().origin)
            self.find_root().pose_graph.add_loop_closure_edge(current,
                                                              match,
                                                              tform_aux.translation[0],  # * self.range_resolution,
                                                              tform_aux.translation[1],  # * self.range_resolution,
                                                              tform_aux.rotation,
                                                              e)
        return a, b, mask, tform, error


class UpdateTformModule(PipelineModule):
    def __init__(self, range_resolution=1):
        super().__init__()
        self.range_resolution = range_resolution

    def run(self, a, b, mask, tform, error):
        print(f"    > Original error: {error}")
        # e = np.array(error) * 0.25
        # e = np.diag(np.power(e, 2))
        # e = np.array([
        #     [10000, 0, 0],
        #     [0, 10000, 0],
        #     [0, 0, 1000000]
        # ])
        e = 1 - np.array(error)
        e = np.diag(np.power(e, 2)) * 1000

        # e = np.linalg.inv(np.diag(np.power(e, 2)))
        # e = np.linalg.inv((error ** 2) * np.eye(3))
        # e = np.diag(np.array(error) * 0.001)
        # e = np.diag(np.array(error))
        print(f"    > "
              f"Translation: {tform.translation}, "
              f"Rotation: {tform.rotation}, "
              f"Scale: {tform.scale}, "
              f"Error: {error}")

        tform_aux = SimilarityTransform(translation=-self.find_root().origin)
        tform_aux += tform
        tform_aux += SimilarityTransform(translation=self.find_root().origin)
        self.find_root().pose_graph.add_odometry(
            tform_aux.translation[0],  # * self.range_resolution,
            tform_aux.translation[1],  # * self.range_resolution,
            tform_aux.rotation,
            e)
        self.find_root().total_tform += tform

        return a, b, mask, tform, error


class OdometerModule(PipelineModule):
    def __init__(self, range_resolution=1, output=None):
        super().__init__()
        self.range_resolution = range_resolution
        self.output = output

        self.node_id = [0]
        self.total_distance = [0.0]
        self.x = [0.0]
        self.y = [0.0]
        self.theta = [0.0]
        self.speed_x = [0.0]
        self.error_x = [0.0]
        self.speed_y = [0.0]
        self.error_y = [0.0]
        self.speed_theta = [0.0]
        self.error_theta = [0.0]

    def __del__(self):
        print("Saving odometer data")
        if self.output is not None:
            print(f"    > Saving to {self.output}")
            np.savetxt(os.path.join(self.output, f"odometer.csv"),
                       np.array([self.node_id, self.x, self.y, self.theta, self.speed_x, self.speed_y, self.speed_theta,
                                 self.total_distance, self.error_x,
                                 self.error_y, self.error_theta]).T, delimiter=",",
                       header="id,Δx,Δy,Δθ,speed_x,speed_y,speed_θ,total_distance,error_x,error_y,error_θ")
            print("    > Saved")

    def run(self, a, b, mask, tform, error):
        if tform is not None:
            self.node_id.append(self.find_root().pose_graph.get_last().id())
            tform_aux = SimilarityTransform(translation=-self.find_root().origin)
            tform_aux += tform
            tform_aux += SimilarityTransform(translation=self.find_root().origin)

            theta = tform.rotation
            x = tform.translation[0] * self.range_resolution
            y = tform.translation[1] * self.range_resolution

            self.x.append(x)
            self.y.append(y)
            self.theta.append(theta)
            self.total_distance.append(self.total_distance[-1] + np.sqrt(x ** 2 + y ** 2))
            print(f"    > Total distance: {self.total_distance[-1]}")

            self.speed_x.append(x / 0.1)
            self.speed_y.append(y / 0.1)
            self.speed_theta.append(theta / 0.1)

            self.error_x.append(error[0])
            self.error_y.append(error[1])
            self.error_theta.append(error[2])
            print(f"    > Speed: X -> {self.speed_x[-1]}, Y -> {self.speed_y[-1]}, θ -> {self.speed_theta[-2]}")

        return a, b, mask, tform, error
