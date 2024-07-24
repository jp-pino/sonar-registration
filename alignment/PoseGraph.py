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

    def add_fixed_pose(self, pose, vertex_id=None):
        '''
        Add fixed pose to the graph
        '''
        v_se2 = g2o.VertexSE2()
        if vertex_id is None:
            vertex_id = self.vertex_count
        v_se2.set_id(vertex_id)
        if self.verbose:
            print("    > PoseGraph: Adding fixed pose vertex with ID", vertex_id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(True)
        self.optimizer.add_vertex(v_se2)
        self.last_id = vertex_id
        self.vertex_count += 1

    def add_vertex(self, id, pose, fixed=False):
        vertex = g2o.VertexSE2()
        vertex.set_id(id)
        vertex.set_estimate(g2o.SE2(pose[0], pose[1], pose[2]))
        vertex.set_fixed(fixed)
        self.optimizer.add_vertex(vertex)

    def add_odometry(self, tx, ty, theta, information):
        odometry_pose = g2o.SE2(tx, ty, theta)

        # Create a new vertex ID
        new_vertex_id = self.last_id + 1
        # Get the last vertex pose
        last_vertex = self.optimizer.vertex(self.last_id)
        last_pose = last_vertex.estimate()

        # Calculate the new pose based on the odometry measurement
        new_pose = last_pose * g2o.SE2(odometry_pose[0], odometry_pose[1], odometry_pose[2])

        # Add the new vertex
        self.add_vertex(new_vertex_id, (new_pose[0], new_pose[1], new_pose[2]))

        # Add the odometry edge
        edge = g2o.EdgeSE2()
        edge.set_vertex(0, self.optimizer.vertex(self.last_id))
        edge.set_vertex(1, self.optimizer.vertex(new_vertex_id))
        edge.set_measurement(g2o.SE2(odometry_pose[0], odometry_pose[1], odometry_pose[2]))
        edge.set_information(information)

        # Add a robust kernel to the edge
        kernel = g2o.RobustKernelHuber()
        edge.set_robust_kernel(kernel)
        self.optimizer.add_edge(edge)

        # Update the last vertex ID
        self.last_id = new_vertex_id
        self.vertex_count += 1
        self.edge_count += 1

        return new_vertex_id

    def add_loop_closure_edge(self, id_start, id_end, northings, eastings, heading, information):
        pose = g2o.SE2(northings, eastings, heading)
        # add edge
        e_se2 = g2o.EdgeSE2()
        e_se2.set_id(self.edge_count)
        e_se2.set_vertex(0, self.vertex(id_start))
        e_se2.set_vertex(1, self.vertex(id_end))
        e_se2.set_measurement(pose)
        e_se2.set_information(information)
        self.optimizer.add_edge(e_se2)
        self.edge_count += 1

    def find_possible_matches(self, node_id, delta_pos, delta_theta):
        matches = []
        vertex = self.vertex(node_id)
        x, y, theta = vertex.estimate().to_vector()
        # Find closes vertices
        for vertex_2 in self.optimizer.vertices().values():
            if vertex.id() == vertex_2.id() or vertex_2.id() == (vertex.id() - 1) or vertex_2.id() == 0:
                continue
            test_x, test_y, test_theta = vertex_2.estimate().to_vector()
            if np.linalg.norm([x - test_x, y - test_y]) <= delta_pos:
                theta_diff = np.abs(theta - test_theta)
                if theta_diff <= np.deg2rad(delta_theta):
                    matches.append(vertex_2.id())
        return matches

    def optimize(self, iterations=10, verbose=None):
        '''
        Optimize the graph
        '''
        self.optimizer.initialize_optimization()
        if verbose is None:
            verbose = self.verbose
        self.optimizer.set_verbose(verbose)
        self.optimizer.optimize(iterations)
        return self.optimizer.chi2()
