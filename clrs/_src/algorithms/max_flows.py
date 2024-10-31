from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
from clrs._src.algorithms import graphs
import numpy as np
import networkx as nx

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

def _ff_impl(A: _Array, s: int, t: int, probes, w):
    f = np.zeros((A.shape[0], A.shape[0]))
    df = np.array(0)

    C = _minimum_cut(A, s, t)

    def reverse(pi):
        u, v = pi[t], t
        while u != v:
            yield u, v
            v = u
            u = pi[u]

    d = np.zeros(A.shape[0])
    msk = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    d[s] = 0
    msk[s] = 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'mask': np.copy(msk),
            'd': np.copy(d),
            'pi_h': np.copy(pi),
            'f_h': np.copy(f),
            'df': np.copy(df),
            'c_h': np.copy(C),
            '__is_bfs_op': np.copy([1])
        })

    while True:
        for _ in range(A.shape[0]):
            prev_d = np.copy(d)
            prev_msk = np.copy(msk)
            for u in range(A.shape[0]):
                for v in range(A.shape[0]):
                    if prev_msk[u] == 1 and A[u, v] - abs(f[u, v]) > 0:
                        if msk[v] == 0 or prev_d[u] + w[u, v] < d[v]:
                            d[v] = prev_d[u] + w[u, v]
                            pi[v] = u
                        msk[v] = 1

            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    'pi_h': np.copy(pi),
                    'd': np.copy(prev_d),
                    'mask': np.copy(msk),
                    'f_h': np.copy(f),
                    'df': np.copy(df),
                    'c_h': np.copy(C),
                    '__is_bfs_op': np.copy([1])
                })

            if np.all(d == prev_d):
                break

        if pi[t] == t:
            break

        df = min([
            A[u, v] - f[u, v]
            for u, v in reverse(pi)
        ])

        for u, v in reverse(pi):
            f[u, v] += df
            f[v, u] -= df

        d = np.zeros(A.shape[0])
        msk = np.zeros(A.shape[0])
        pi = np.arange(A.shape[0])
        d[s] = 0
        msk[s] = 1
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pi_h': np.copy(pi),
                'd': np.copy(d),
                'mask': np.copy(msk),
                'f_h': np.copy(f),
                'df': np.copy(df),
                'c_h': np.copy(C),
                '__is_bfs_op': np.array([0])
            })

    return f, probes


def ford_fulkerson(A: _Array, s: int, t: int):

    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['ford_fulkerson'])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'w': np.copy(w),
        })

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'f': np.copy(f)
        }
    )
    probing.finalize(probes)

    return f, probes


def ford_fulkerson_mincut(A: _Array, s: int, t: int):
    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['ford_fulkerson_mincut'])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'w': np.copy(w)
        })

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'f': np.copy(f),
            'c': _minimum_cut(A, s, t)
        }
    )

    probing.finalize(probes)

    return f, probes


def _minimum_cut(A, s, t):
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


if __name__ == "__main__":
    # Test the Edmonds-Karp algorithm on a simple graph
    DIRECTED_WEIGHTED_GRAPH = np.array([
        [0, 10, 0, 10, 0, 0],
        [0, 0, 4, 2, 8, 0],
        [0, 0, 0, 0, 0, 10],
        [0, 0, 9, 0, 0, 10],
        [0, 0, 0, 6, 0, 10],
        [0, 0, 0, 0, 0, 0],
    ])

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

    max_flow, _ = ford_fulkerson(DIRECTED_WEIGHTED_GRAPH, 0, 5)
    print("Ford fulkerson, result matrix:")
    print(max_flow)

    cut = _minimum_cut(DIRECTED_WEIGHTED_GRAPH, 0, 5)
    print("Minimum cut:")
    print(cut)

    ff_mincut, _ = ford_fulkerson_mincut(DIRECTED_WEIGHTED_GRAPH, 0, 5)
    print("Ford fulkerson mincut, result matrix:")
    print(ff_mincut)

    cut = _minimum_cut(UNDIRECTED_UNIFORM_GRAPH, 0, 4)
    print("Minimum cut undir unif:")
    print(cut)

    cut = _minimum_cut(UNDIRECTED_WEIGHTED_GRAPH, 0, 4)
    print("Minimum cut undir wei:")
    print(cut)

    cut = _minimum_cut(DIRECTED_UNIFORM_GRAPH, 0, 5) # unreachable
    print("Minimum cut unreachable dir unif:")
    print(cut)

    cut = _minimum_cut(DIRECTED_UNIFORM_GRAPH, 0, 4)
    print("Minimum cut dir unif:")
    print(cut)