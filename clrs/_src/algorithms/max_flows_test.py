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
# Partition: ({0}, {1, 2, 3, 4})

UNDIRECTED_WEIGHTED_GRAPH = np.array([
    [0, 2, 3, 0, 0],
    [2, 0, 1, 3, 2],
    [3, 1, 0, 0, 1],
    [0, 3, 0, 0, 5],
    [0, 2, 1, 5, 0],
])
# Partition: ({0, 2}, {1, 3, 4})

DIRECTED_UNIFORM_GRAPH = np.array([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
])
# Partition: ({0, 1, 3, 4}, {2, 5})

DIRECTED_WEIGHTED_GRAPH = np.array([
    [0, 10, 0, 10, 0, 0],
    [0, 0, 4, 2, 8, 0],
    [0, 0, 0, 0, 0, 10],
    [0, 0, 9, 0, 0, 10],
    [0, 0, 0, 6, 0, 10],
    [0, 0, 0, 0, 0, 0],
])
# Partition: ({0}, {1, 2, 3, 4, 5})

CAPACITY_MATRIX_1 = np.array([
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0],
])
# Partition: ({0, 1, 2, 4}, {3, 5})


CAPACITY_MATRIX_2 = np.array([
    [0, 3, 2, 0],
    [0, 0, 5, 2],
    [0, 0, 0, 3],
    [0, 0, 0, 0],
])
# Partition: ({0}, {1, 2, 3})
# manually, I got a diff partition
# both nx min cut and my implementation agree on this one tho

class MaxFlowsTest(absltest.TestCase):

    def convert_min_cut_to_array(self, set_reachable, set_unreachable, num_nodes):
        cut_array = np.zeros(num_nodes)
        cut_array[list(set_reachable)] = 1
        return cut_array

    def _minimum_cut(self, A, s, t):
        graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
        for i, j in zip(*A.nonzero()):
            graph[i][j]['capacity'] = A[i, j]
        cut_value, partition = nx.minimum_cut(graph, s, t)
        return partition

    def test_max_flow_min_cut_undirected_uniform(self):
        capacity = UNDIRECTED_UNIFORM_GRAPH
        expected_cut = np.array([1, 0, 0, 0, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 4)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_undirected_weighted(self):
        capacity = UNDIRECTED_WEIGHTED_GRAPH
        expected_cut = np.array([1, 0, 1, 0, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 4)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_directed_uniform(self): # Correct
        capacity = DIRECTED_UNIFORM_GRAPH
        expected_cut = np.array([1, 1, 0, 1, 1, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 5)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_directed_weighted(self):
        capacity = DIRECTED_WEIGHTED_GRAPH
        expected_cut = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 5)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_capacity_1(self):
        capacity = CAPACITY_MATRIX_1
        expected_cut = np.array([1, 1, 1, 0, 1, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 5)
        np.testing.assert_array_equal(expected_cut, cut)

    def test_max_flow_min_cut_capacity_2(self):
        capacity = CAPACITY_MATRIX_2
        expected_cut = np.array([1, 0, 0, 0], dtype=float)
        cut, _ = max_flows.max_flow_min_cut(capacity, 0, 3)
        np.testing.assert_array_equal(expected_cut, cut)

if __name__ == "__main__":
    absltest.main()