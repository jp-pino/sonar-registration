import os

import g2o
import matplotlib.pyplot as plt
import numpy as np
import ray
from skimage.transform import SimilarityTransform

from pipeline.PipelineModule import PipelineModule
from alignment.PoseGraph import PoseGraph


class FindNeighborsModule(PipelineModule):
    def __init__(self, registration_id, delta_theta=40, delta_radius=35, error_threshold=0.8, range_resolution=1,
                 output=None, matches_number=15):
        super().__init__()
        self.registration_id = registration_id
        self.delta_theta = delta_theta
        self.delta_radius = delta_radius
        self.error_threshold = error_threshold
        self.output = output
        self.range_resolution = range_resolution
        self.log = []
        self.matches_number = matches_number

    def __del__(self):
        if self.output is not None and len(self.log) > 0:
            print("Saving log")
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
        size = len(matches) if len(matches) < self.matches_number else self.matches_number
        # matches = np.random.choice(matches, size, replace=False).tolist()

        # Select N best matches
        matches = matches[:size]

        # Select previous 5 frames if available
        for i in [current - i for i in range(2, 6 if current > 5 else current)]:
            if i not in matches:
                matches.append(i)

        print(f"    > Selected {len(matches)} possible matches: {matches}")

        # Remove id 0 and 1 if present
        for i in [0]:
            if i in matches:
                matches.remove(i)

        for match in matches:
            print(f"    > Processing match {current} - {match}")
            print(f"    > Source cache length: {len(self.find_root().get_module(self.registration_id).source_cache)}")
            _, b_match, mask_match = self.find_root().get_module(self.registration_id).read_cache(match - 1)
            b_processed, b_match_processed, _, tform_neighbor, error_neighbor \
                = self.find_root().get_module(self.registration_id).run(b.copy(),
                                                                        b_match.copy(),
                                                                        mask,
                                                                        SimilarityTransform(),
                                                                        [0, 0, 0])

            if (error_neighbor[0] > self.error_threshold
                    or error_neighbor[1] > self.error_threshold
                    or error_neighbor[2] > self.error_threshold):
                print(f"    > Skipping match due to high error: {error_neighbor}")
                continue

            print(f"    > Original error: {error_neighbor}")

            e = np.power(1 / (np.array(error_neighbor) * 10000), 2)

            print(f"    > "
                  f"Match ({current} - {match}) "
                  f"Translation: {tform_neighbor.translation}, "
                  f"Rotation: {tform_neighbor.rotation}, "
                  f"Scale: {tform_neighbor.scale}, "
                  f"Error: {e}")

            # if np.max(np.abs(e)) > 10:
            #     print(f"    > Skipping match due to high error: {error_neighbor}")
            #     continue

            # if self.output is not None:
            # plt.imsave(os.path.join(self.output, f"match_{current}_{match}_a.png"), a, cmap="gray")
            # plt.imsave(os.path.join(self.output, f"match_{current}_{match}_b.png"), b, cmap="gray")
            # plt.imsave(os.path.join(self.output, f"match_{current}_{match}_match.png"), b_match, cmap="gray")
            # plt.imsave(os.path.join(self.output, f"match_{current}_{match}_b_match.png"), b_match_processed, cmap="gray")
            # plt.imsave(os.path.join(self.output, f"match_{current}_{match}_b_processed.png"), b_processed, cmap="gray")

            self.log.append(
                [current, match, tform_neighbor.translation[0], tform_neighbor.translation[1], tform_neighbor.rotation,
                 error_neighbor[0], error_neighbor[1], error_neighbor[2]])

            self.find_root().pose_graph.add_edge(current,
                                                 match,
                                                 g2o.SE2(tform_neighbor.translation[0],
                                                         tform_neighbor.translation[1],
                                                         tform_neighbor.rotation),
                                                 np.diag(e))
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
        # e = 1 - np.array(error)
        # e = np.diag(np.power(e, 2)) * 1000

        e = np.power(1 / (np.array(error) / 1000), 2)

        # e = np.linalg.inv(np.diag(np.power(e, 2)))
        # e = np.linalg.inv((error ** 2) * np.eye(3))
        # e = np.diag(np.array(error) * 0.001)
        # e = np.diag(np.array(error))

        print(f"    > "
              f"Translation: {tform.translation}, "
              f"Rotation: {tform.rotation}, "
              f"Scale: {tform.scale}, "
              f"Error: {e}")

        self.find_root().pose_graph.add_odometry(
            g2o.SE2(tform.translation[0],
                    tform.translation[1],
                    tform.rotation),
            np.diag(e))
        self.find_root().total_tform += tform

        return a, b, mask, tform, error


class OdometerModule(PipelineModule):
    def __init__(self, range_resolution=0.0025, output=None):
        super().__init__()
        self.range_resolution = range_resolution
        self.output = output

        self.node_id = []
        self.distance_delta = []
        self.total_distance = []
        self.distance_from_start = []
        self.x = []
        self.y = []
        self.theta = []
        self.speed_x = []
        self.error_x = []
        self.speed_y = []
        self.error_y = []
        self.speed_theta = []
        self.error_theta = []

    def __del__(self):
        if self.output is not None and len(self.node_id) > 1:
            print("Saving odometer data")
            print(f"    > Saving to {self.output}")
            self.save_file("odometer.csv")
            print("    > Saved")

            self.node_id = []
            self.total_distance = []
            self.x = []
            self.y = []
            self.theta = []
            self.speed_x = []
            self.speed_y = []
            self.speed_theta = []
            print("Generating realigned odometry")
            total_tform = SimilarityTransform()
            for vertex in reversed(self.find_root().pose_graph.optimizer.vertices().values()):
                if type(vertex) != g2o.VertexSE2:
                    continue
                if vertex.id() == 0:
                    continue
                self.node_id.append(vertex.id())
                x, y, theta = vertex.estimate().to_vector()
                tform = SimilarityTransform(translation=[x, y], rotation=theta)
                total_tform += tform
                self.extract_data(tform, total_tform)
            self.save_file("realigned_odometer.csv")
            print("    > Saved")

    def save_file(self, path):
        np.savetxt(os.path.join(self.output, path),
                   np.array(
                       [self.node_id, self.x, self.y, self.theta, self.speed_x, self.speed_y, self.speed_theta,
                        self.total_distance, self.distance_from_start, self.error_x,
                        self.error_y, self.error_theta]).T, delimiter=",",
                   header="id,Δx,Δy,Δθ,speed_x,speed_y,speed_θ,total_distance,distance_from_start,error_x,error_y,error_θ")

    def extract_data(self, tform, total_tform):

        theta = tform.rotation
        print(f"    > Odom range resolution: {self.range_resolution}")
        x = tform.translation[0] * self.range_resolution
        y = tform.translation[1] * self.range_resolution

        self.x.append(x)
        self.y.append(y)
        self.theta.append(theta)

        if len(self.total_distance) == 0:
            self.total_distance.append(np.sqrt(x ** 2 + y ** 2))
        else:
            self.total_distance.append(self.total_distance[-1] + np.sqrt(x ** 2 + y ** 2))
        print(f"    > Odom Total distance: {self.total_distance[-1]}")

        self.speed_x.append(x / 0.1)
        self.speed_y.append(y / 0.1)
        self.speed_theta.append(theta / 0.1)

        x = total_tform.translation[0] * self.range_resolution
        y = total_tform.translation[1] * self.range_resolution
        self.distance_from_start.append(np.sqrt(x ** 2 + y ** 2))

    def run(self, a, b, mask, tform, error):
        if tform is not None:
            print(f"    > Odom tform: {tform.translation}, {tform.rotation}")
            self.node_id.append(self.find_root().pose_graph.get_last().id())

            tform_aux = SimilarityTransform(translation=-self.find_root().origin)
            tform_aux += tform
            tform_aux += SimilarityTransform(translation=self.find_root().origin)

            total_tform = SimilarityTransform(translation=-self.find_root().origin)
            total_tform += self.find_root().total_tform
            total_tform += SimilarityTransform(translation=self.find_root().origin)

            self.extract_data(tform_aux, total_tform)

            self.error_x.append(error[0])
            self.error_y.append(error[1])
            self.error_theta.append(error[2])
            if len(self.x) > 1:
                print(f"    > Odom Speed: X -> {self.speed_x[-1]}, Y -> {self.speed_y[-1]}, θ -> {self.speed_theta[-2]}")

        return a, b, mask, tform, error
