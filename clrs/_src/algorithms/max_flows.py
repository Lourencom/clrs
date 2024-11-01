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
            '__is_bfs_op': np.array([1])
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
                    '__is_bfs_op': np.array([1])
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
    """
    This is exactly the same as the Ford-Fulkerson algorithm, but with an additional
    output probe for the minimum cut.
    Also, insinde the _ff_impl function, the minimum cut is also passed as hint.
    """
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


def max_flow_min_cut(capacity: _Array, s: int, t: int) -> _Out:
    """Edmonds-Karp algorithm for computing max-flow and min-cut."""

    chex.assert_rank(capacity, 2)
    probes = probing.initialize(specs.SPECS['max_flow_min_cut'])

    num_nodes = capacity.shape[0]
    pos = np.arange(num_nodes)

    # Push input probes
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(pos) * 1.0 / num_nodes,
            's': probing.mask_one(s, num_nodes),
            't': probing.mask_one(t, num_nodes),
            'capacity': np.copy(capacity),
            'adj': probing.graph(np.copy(capacity))
        })

    # Initialize flow and residual capacities
    flow = np.zeros_like(capacity)
    residual_capacity = np.copy(capacity)

    while True:
        # Initialize BFS for finding augmenting path
        visited = np.zeros(num_nodes, dtype=bool)
        parent = -np.ones(num_nodes, dtype=int)
        queue = [s]
        visited[s] = True

        current_cut_h = visited.astype(float) # fixme: not sure if i like this hack

        # Push initial hint probes for BFS
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'flow_h': np.copy(flow),
                'residual_capacity_h': np.copy(residual_capacity),
                'visited_h': visited.astype(float),
                'path_h': np.copy(parent),
                'augmenting_path_h': np.zeros(num_nodes),
                'u_h': probing.mask_one(s, num_nodes),
                'current_cut_h': np.copy(current_cut_h), # fixme: not sure if i like this hack
            })

        found_augmenting_path = False

        # BFS loop
        while queue:
            u = queue.pop(0)
            for v in range(num_nodes):
                if not visited[v] and residual_capacity[u, v] > 0:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)

                    current_cut_h = visited.astype(float)  # fixme: not sure if i like this hack

                    # Push hint probes during BFS
                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            'flow_h': np.copy(flow),
                            'residual_capacity_h': np.copy(residual_capacity),
                            'visited_h': visited.astype(float),
                            'path_h': np.copy(parent),
                            'augmenting_path_h': np.zeros(num_nodes),
                            'u_h': probing.mask_one(u, num_nodes),
                            'current_cut_h': np.copy(current_cut_h),  # fixme: not sure if i like this hack
                        })

                    if v == t:
                        found_augmenting_path = True
                        break
            if found_augmenting_path:
                break

        # If no augmenting path is found, exit the loop
        if not found_augmenting_path:
            break

        # Trace back the augmenting path and find bottleneck capacity
        path = []
        v = t
        bottleneck = float('inf')
        while v != s:
            u = parent[v]
            bottleneck = min(bottleneck, residual_capacity[u, v])
            path.append(v)
            v = u
        path.append(s)
        path = path[::-1]  # Reverse to get path from source to sink

        # Create mask for nodes in the augmenting path
        augmenting_path_mask = np.zeros(num_nodes)
        augmenting_path_mask[path] = 1.0

        # Update flow and residual capacities along the augmenting path
        v = t
        while v != s:
            u = parent[v]
            flow[u, v] += bottleneck
            flow[v, u] -= bottleneck  # Reverse flow for residual graph
            residual_capacity[u, v] -= bottleneck
            residual_capacity[v, u] += bottleneck
            v = u

        # Compute current_cut_h after flow update fixme fixme this is hack too
        visited_cut = np.zeros(num_nodes, dtype=bool)
        queue_cut = [s]
        visited_cut[s] = True
        while queue_cut:
            u_cut = queue_cut.pop(0)
            for v_cut in range(num_nodes):
                if not visited_cut[v_cut] and residual_capacity[u_cut, v_cut] > 0:
                    visited_cut[v_cut] = True
                    queue_cut.append(v_cut)
        current_cut_h = visited_cut.astype(float)

        # Push hint probes after updating flow and residual capacities
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'flow_h': np.copy(flow),
                'residual_capacity_h': np.copy(residual_capacity),
                'visited_h': visited.astype(float),
                'path_h': np.copy(parent),
                'augmenting_path_h': np.copy(augmenting_path_mask),
                'u_h': probing.mask_one(u, num_nodes),
                'current_cut_h': np.copy(current_cut_h),  # fixme: not sure if i like this hack
            })

    """
    # After max-flow computation, find the min-cut
    # Perform BFS to find reachable nodes from source in residual graph
    visited = np.zeros(num_nodes, dtype=bool)
    queue = [s]
    visited[s] = True
    while queue:
        u = queue.pop(0)
        for v in range(num_nodes):
            if not visited[v] and residual_capacity[u, v] > 0:
                visited[v] = True
                queue.append(v)

    # Nodes reachable from source are on one side of the min-cut
    cut = visited.astype(float)
    """

    cut = current_cut_h # fixme: not sure if i like this hack

    # Push output probes
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'cut': np.copy(cut)
        })

    probing.finalize(probes)

    return cut, probes
