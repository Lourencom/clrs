from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
from clrs._src.algorithms import graphs
import numpy as np

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]

def edmonds_karp(A: _Array, s: int, t: int) -> _Out:
    """Edmonds-Karp max-flow algorithm (Ford-Fulkerson method using BFS)."""

    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['edmonds_karp'])  # Updated to use 'edmonds_karp' spec

    A_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            'd': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
        })

    # Initialize residual capacity matrix and other necessary variables
    residual = np.copy(A)
    flow = np.zeros(A.shape)  # To track flow along edges
    mark = np.zeros(A.shape[0])  # To mark visited nodes
    pi = np.full(A.shape[0], -1)  # Predecessor array

    def bfs(source, sink):
        """BFS to find augmenting path."""
        queue = [source]
        mark[:] = 0
        mark[source] = 1
        pi[:] = -1
        while queue:
            u = queue.pop(0)
            for v in range(A.shape[0]):
                if residual[u, v] > 0 and not mark[v]:  # Check for residual capacity
                    queue.append(v)
                    mark[v] = 1
                    pi[v] = u
                    if v == sink:
                        return True
        return False

    while bfs(s, t):
        # Find the maximum flow through the path found by BFS
        path_flow = float('Inf')
        v = t
        while v != s:
            u = pi[v]
            path_flow = min(path_flow, residual[u, v])
            v = u

        # Update residual capacities of the edges and reverse edges
        v = t
        while v != s:
            u = pi[v]
            residual[u, v] -= path_flow
            residual[v, u] += path_flow
            flow[u, v] += path_flow
            v = u

        # Push current state to the probes (intermediate state tracking)
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'flow': np.copy(flow),
                'residual': np.copy(residual),
                'pi_h': np.copy(pi),
                'mark': np.copy(mark),
                'u': probing.mask_one(u, A.shape[0]),  # Current vertex
                'v': probing.mask_one(v, A.shape[0]),  # Next vertex
                'cut_h': np.copy((residual == 0) & (A > 0))  # Intermediate hint for the min-cut
            })

    # The min-cut is the set of edges with residual capacity == 0
    cut = (residual == 0) & (A > 0)

    # Push the final output (min-cut and predecessor pointers)
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'cut': np.copy(cut),
            'pi': np.copy(pi),  # Predecessor pointers for nodes
        })

    probing.finalize(probes)

    return cut, probes

