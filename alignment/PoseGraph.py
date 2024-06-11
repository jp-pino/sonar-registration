import g2o
import numpy as np


class PoseGraph:
    def __init__(self, verbose=False) -> None:
        '''
        GraphSLAM in 2D with G2O
        '''
        self.optimizer = g2o.SparseOptimizer()
        self.solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
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
        self.vertex_count += 1

    def add_odometry(self, northings, eastings, heading, information=np.eye(3), invert=False):
        '''
        Add odometry to the graph
        '''
        # Find the last pose vertex id
        vertices = self.optimizer.vertices()
        if len(vertices) == 0:
            raise ValueError("There is no previous pose, have you forgot to add a fixed initial pose?")
        if self.last_id is None:
            self.last_id = [v for v in vertices if type(vertices[v]) == g2o.VertexSE2][0]
        v_se2 = g2o.VertexSE2()
        if self.verbose:
            print("    > PoseGraph: Adding pose vertex", self.vertex_count)
        v_se2.set_id(self.vertex_count)
        pose = g2o.SE2(northings, eastings, heading)
        if self.verbose:
            print(f"    > PoseGraph: Adding odometry with pose: {pose.to_vector()}")
        if invert:
            pose = pose.inverse()
        v_se2.set_estimate(self.vertex_pose(self.last_id) * pose)
        self.optimizer.add_vertex(v_se2)
        # add edge
        e_se2 = g2o.EdgeSE2()
        e_se2.set_id(self.edge_count)
        e_se2.set_vertex(0, self.vertex(self.last_id))
        e_se2.set_vertex(1, self.vertex(self.vertex_count))
        e_se2.set_measurement(pose)
        e_se2.set_information(information)
        id = self.vertex_count
        self.optimizer.add_edge(e_se2)
        self.vertex_count += 1
        self.edge_count += 1
        if self.verbose:
            print("    > PoseGraph: Adding SE2 edge between", self.last_id, id)
        self.last_id = id
        return id

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
