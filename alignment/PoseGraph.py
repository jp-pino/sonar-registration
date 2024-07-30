import g2o
import numpy as np


class PoseGraph:
    def __init__(self, verbose=False) -> None:
        '''
        GraphSLAM in 2D with G2O
        '''
        self.optimizer = g2o.SparseOptimizer()
        self.solver = g2o.BlockSolverSE2(g2o.LinearSolverPCGSE2())
        self.algorithm = g2o.OptimizationAlgorithmLevenberg(self.solver)
        self.optimizer.set_algorithm(self.algorithm)

        self.vertex_count = 0
        self.edge_count = 0
        self.verbose = verbose
        self.last_id = None

    def get_last(self):
        return self.vertex(self.last_id)

    def vertex_pose(self, id):
        '''
        Get position of vertex by id
        '''
        return self.optimizer.vertex(id).estimate()

    def vertex(self, id):
        '''
        Get vertex by id
        '''
        return self.optimizer.vertex(id)

    def add_vertex(self, pose, set_last=False, fixed=False):
        vertex_id = self.vertex_count
        vertex = g2o.VertexSE2()
        vertex.set_id(vertex_id)
        vertex.set_estimate(pose)
        vertex.set_fixed(fixed)
        self.optimizer.add_vertex(vertex)
        self.vertex_count += 1
        if set_last:
            self.last_id = vertex_id
        return vertex

    def add_edge(self, id_start, id_end, pose, information):
        print(f"Adding edge from {id_start} to {id_end}")
        edge = g2o.EdgeSE2()
        edge_id = self.edge_count
        edge.set_id(edge_id)
        edge.set_vertex(0, self.vertex(id_start))
        edge.set_vertex(1, self.vertex(id_end))
        edge.set_measurement(pose)
        edge.set_information(information)
        edge.set_robust_kernel(g2o.RobustKernelHuber())
        self.optimizer.add_edge(edge)
        self.edge_count += 1
        return edge

    def add_odometry(self, odometry_pose, information, boost=1):
        last_id = self.last_id

        # Get the last vertex pose
        last_pose = self.optimizer.vertex(last_id).estimate()

        # Calculate the new pose based on the odometry measurement
        new_pose = last_pose * odometry_pose

        # Add the new vertex
        vertex = self.add_vertex(new_pose, set_last=True)

        # Add the odometry edge
        for i in range(boost):
            self.add_edge(last_id, vertex.id(), odometry_pose, information)

        return vertex

    def add_landmark_to_last_vertex(self, x, y):
        # Create a new vertex ID
        landmark_id = self.vertex_count + 1

        # Create vertex
        vertex = g2o.VertexPointXY()
        vertex.set_id(landmark_id)
        vertex.set_estimate(np.array([x, y]))
        vertex.set_fixed(True)
        self.optimizer.add_vertex(vertex)

        # Create edge
        edge = g2o.EdgeSE2PointXY()
        edge.set_id(self.edge_count)
        edge.set_vertex(0, self.get_last())
        edge.set_vertex(1, self.optimizer.vertex(landmark_id))
        edge.set_measurement(np.array([0, 0]))
        edge.set_information(np.diag(np.array([1000, 1000])))
        self.optimizer.add_edge(edge)

        print(f"Landmark id {landmark_id}")

        self.vertex_count += 1
        self.edge_count += 1

        return vertex

    def find_possible_matches(self, node_id, delta_pos, delta_theta):
        matches = []
        vertex = self.vertex(node_id)
        x, y, theta = vertex.estimate().to_vector()
        # Find closes vertices
        for vertex_2 in self.optimizer.vertices().values():
            if type(vertex_2) != g2o.VertexSE2:
                continue
            if vertex.id() == vertex_2.id() or vertex_2.id() == 0:
                continue
            test_x, test_y, test_theta = vertex_2.estimate().to_vector()
            distance = np.linalg.norm([x - test_x, y - test_y])
            if distance <= delta_pos:
                theta_diff = np.abs(theta - test_theta)
                if theta_diff <= np.deg2rad(delta_theta):
                    matches.append([vertex_2.id(), distance, theta_diff])

        # Sort by distance and theta difference
        matches = sorted(matches, key=lambda el: el[1] * 10 + el[2])

        # Return only the vertex IDs
        matches = [el[0] for el in matches]

        return matches

    def optimize(self, iterations=10, verbose=None):
        '''
        Optimize the graph
        '''
        self.optimizer.initialize_optimization()
        if verbose is None:
            verbose = self.verbose
        self.optimizer.set_verbose(verbose)
        # self.optimizer.save("test.g2o")
        self.optimizer.optimize(iterations)
        return self.optimizer.chi2()
