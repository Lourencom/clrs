from absl.testing import absltest
import numpy as np
from clrs._src.algorithms import max_flows

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

class MaxFlowsTest(absltest.TestCase):

    def test_edmonds_karp_undirected_uniform(self):
        """Test Edmonds-Karp on an undirected graph with uniform edge capacities."""
        expected_max_flow = 2
        flow, _ = max_flows.edmonds_karp(UNDIRECTED_UNIFORM_GRAPH, 0, 4)
        total_flow = np.sum(flow[0, :]) # Compute total flow from source to sink
        self.assertEqual(total_flow, expected_max_flow)

    def test_edmonds_karp_undirected_weighted(self):
        """Test Edmonds-Karp on an undirected graph with varying edge capacities."""
        expected_max_flow = 4
        flow, _ = max_flows.edmonds_karp(UNDIRECTED_WEIGHTED_GRAPH, 0, 4)

        total_flow = np.sum(flow[0, :]) # Compute total flow from source to sink
        self.assertEqual(total_flow, expected_max_flow)

    def test_edmonds_karp_directed_uniform(self):
        """Test Edmonds-Karp on a directed graph with uniform edge capacities."""
        expected_max_flow = 1
        flow, _ = max_flows.edmonds_karp(DIRECTED_UNIFORM_GRAPH, 0, 4)

        total_flow = np.sum(flow[0, :])  # Compute total flow from source to sink
        self.assertEqual(total_flow, expected_max_flow)

    def test_edmonds_karp_non_reachable(self):
        """Test Edmonds-Karp on a directed graph with uniform edge capacities."""
        expected_max_flow = 0 # Just to test the case where Node 5 is not reachable from 0
        flow, _ = max_flows.edmonds_karp(DIRECTED_UNIFORM_GRAPH, 0, 5)

        total_flow = np.sum(flow[0, :])  # Compute total flow from source to sink
        self.assertEqual(total_flow, expected_max_flow)

    def test_edmonds_karp_directed_weighted(self):
        """Test Edmonds-Karp on a directed graph with varying edge capacities."""
        expected_max_flow = 20
        flow, _ = max_flows.edmonds_karp(DIRECTED_WEIGHTED_GRAPH, 0, 5)
        total_flow = np.sum(flow[0, :]) # Compute total flow from source to sink
        self.assertEqual(total_flow, expected_max_flow)

    

if __name__ == "__main__":
    absltest.main()
