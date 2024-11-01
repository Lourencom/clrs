from absl.testing import absltest
import numpy as np
from clrs._src.algorithms import max_flows
import networkx as nx
# Max-flow test cases

# Undirected with uniform capacities
UNDIRECTED_UNIFORM_GRAPH = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
])

# Undirected with weighted capacities
UNDIRECTED_WEIGHTED_GRAPH = np.array([
    [0, 2, 3, 0, 0],
    [2, 0, 1, 3, 2],
    [3, 1, 0, 0, 1],
    [0, 3, 0, 0, 5],
    [0, 2, 1, 5, 0],
])

# Directed with uniform capacities
DIRECTED_UNIFORM_GRAPH = np.array([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
])

# Directed with weighted capacities
DIRECTED_WEIGHTED_GRAPH = np.array([
    [0, 10, 0, 10, 0, 0],
    [0, 0, 4, 2, 8, 0],
    [0, 0, 0, 0, 0, 10],
    [0, 0, 9, 0, 0, 10],
    [0, 0, 0, 6, 0, 10],
    [0, 0, 0, 0, 0, 0],
])

CAPACITY_MATRIX_1 = np.array([
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0],
])

CAPACITY_MATRIX_2 = np.array([
    [0, 3, 2, 0],
    [0, 0, 5, 2],
    [0, 0, 0, 3],
    [0, 0, 0, 0],
])

class MaxFlowsTest(absltest.TestCase):

    def _minimum_cut(self, A, s, t):
        C = np.zeros((A.shape[0], 2))

        graph = nx.from_numpy_array(A)
        nx.set_edge_attributes(graph, {(i, j): A[i, j] for i, j in zip(*A.nonzero())},
                               name='capacity')

        _, cuts = nx.minimum_cut(graph, s, t)

        for v in cuts[0]:
            C[v][0] = 1

        for v in cuts[1]:
            C[v][1] = 1

        return C

    def test_max_flow_min_cut_1(self):
        # Test Case 1
        capacity = CAPACITY_MATRIX_1
        expected_cut = np.array([1, 1, 1, 0, 1, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 5)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_2(self):
        # Test Case 2
        capacity = CAPACITY_MATRIX_2
        expected_cut = np.array([1, 0, 0, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 3)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_3(self):
        # Test Case 3 (Directed graph)
        capacity = np.array([
            [0, 10, 10, 0],
            [0, 0, 2, 4],
            [0, 0, 0, 8],
            [0, 0, 0, 0],
        ])
        expected_cut = np.array([1, 1, 1, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 3)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_4(self):
        # Test Case 4 (Disconnected graph)
        capacity = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        expected_cut = np.array([1, 0, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 2)
        np.testing.assert_array_equal(expected_cut, cut)


if __name__ == "__main__":
    absltest.main()