import numpy as np
from scipy.linalg import fractional_matrix_power
from igraph import VertexClustering, VertexDendrogram
from igraph.drawing.colors import ClusterColoringPalette

def _get_memberships(a, b):
    i = 0
    j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(0)
            i += 1
        else:
            result.append(1)
            j += 1
    if i < len(a):
        result.extend([0] * (len(a) - i))
    elif j < len(b):
        result.extend([1] * (len(b) - j))

    return result

def spectral_bisection(g):
    # build laplacian matrix
    n_nodes = len(g.vs)
    laplacian = np.zeros(shape=(n_nodes, n_nodes))
    for v in g.vs:
        for n in v.neighbors():
            laplacian[v.index][n.index] = -1
        laplacian[v.index][v.index] = v.degree()

    # eigenvector decomposition of symetric laplacian
    values, vectors = np.linalg.eigh(laplacian)

    # find second smallest eigenvalue
    eval_2 = np.partition(values.flatten(), 2)[2]

    # take median of corresponding eigenvector
    ev_idx = list(values).index(eval_2)
    evec_2 = vectors[ev_idx]
    evec_med = np.median(evec_2)

    # partition g into a and b, a has value <= median, b has value > median
    a = [i for i, v in enumerate(evec_2) if v < evec_med]
    z = [i for i, v in enumerate(evec_2) if v == evec_med]
    b = [i for i, v in enumerate(evec_2) if v > evec_med]
    to_move = []

    # if |a - b| > 1, move elements whose value == median into b
    if len(a) + len(z) - len(b) > 1:
        num_to_move = int(len(medians) / 2) - len(b)
        to_move = list(np.random.choice(z, size=num_to_move, replace=False))

    b.extend(to_move)
    a.extend(list(set(z) - set(to_move)))

    # take vertices that have edges running between a and b
    edge_separator = g.es.select(_between=(a, b))
    g2 = edge_separator.subgraph()
    s1 = set()
    for e in edge_separator:
        t = e.tuple
        for i in t:
            s1.add(g.vs[i]['name'])
    s2 = set(map(lambda v: v['name'], g2.vs))

    # find min vertex cover of vertices
    types = [v['name'] in set([g.vs[i]['name'] for i in a]) for v in g2.vs]
    matching = g2.maximum_bipartite_matching(types=types)

    rm = (set(), set())
    for e in matching.edges():
        idx_to_use = np.random.randint(2)
        rm[idx_to_use].add(g2.vs[e.tuple[idx_to_use]]['name'])

    partitioned = g.copy()
    palette = ClusterColoringPalette(2)
    partitioned.vs['color'] = palette.get_many(_get_memberships(a, b))
    #rm_vs = g.vs.select(lambda v: v['name'] in rm[0] or v['name'] in rm[1])
    #partitioned.delete_vertices(rm_vs)
    
    return partitioned
    
# def edge_betweenness(g):

def modularity(g):
    # build transition matrix
    n_nodes = len(g.vs)
    n_edges = len(g.es)
    adjacent = np.zeros(shape=(n_nodes, n_nodes))
    for v in g.vs:
        for n in v.neighbors():
            adjacent[v.index][n.index] = 1 - (v.degree() * n.degree() / (2 * n_edges))

    # needed?
    # quality =

    # use spectral approach to divide nodes

def _walktrap_node_dist(i, j, probs, t=1):
    return np.linalg.norm(np.subtract(probs[i], probs[j]) * t @ probs)

def _walktrap_com_prob(a, probs, t):
    p = np.matrix([probs[i] for i in a])
    return np.array([np.sum(p[: ,i]) for i in range(p.shape[1])]) * t @ probs / len(a)

def _walktrap_com_dist(a, b, probs, t=1):
    a_prob = _walktrap_com_prob(a, probs, t)
    b_prob = _walktrap_com_prob(b, probs, t)
    return np.linalg.norm(np.subtract(a_prob, b_prob))

def _walktrap_com_var(a, b, probs, t=1):
    dist = _walktrap_com_dist(a, b, probs, t)
    return len(a) * len(b) / (len(a) + len(b)) / probs.shape[0] * (dist ** 2)

def _build_membership_list(idx_to_com, n_nodes):
    mem_count = 0
    memberships = [-1 for _ in range(n_nodes)]
    for n in range(n_nodes):
        if memberships[n] == -1:
            com = idx_to_com[n]
            for v in com:
                memberships[v] = mem_count
            mem_count += 1

    return memberships

def _calculate_mod_quality(g, coms):
    q = 0
    for c in list(coms):
        tot = len(g.es)
        e_in = len(g.es.select(_between=(c, c)))
        outside_c = list(set(range(len(g.vs))) - set(c))
        e_out = len(g.es.select(_between=(c, outside_c)))
        q += e_in / tot - ((e_out / tot) ** 2)

    return q

def _walktrap_solver(g, probs, t, n_clusters, dist_f):
    n_nodes = len(g.vs)

    # initialize n communities with 1 element
    coms = set([(i,) for i in range(len(g.vs))])
    idx_to_com = {i:(i,) for i in range(len(g.vs))}

    # calculate distances between nodes, add placeholders
    dists = np.fromfunction(np.vectorize(dist_f), (n_nodes, n_nodes), dtype=int)
    dists = np.append(dists, np.full((n_nodes, n_nodes), np.inf), axis=1)
    dists = np.append(dists, np.full((n_nodes, n_nodes * 2), np.inf), axis=0)

    # iteratively merge communities that are closest, recompute distance
    merges = []
    modularities = []
    ignore = set()
    memberships = []
    while len(coms) > 1:
        # find communities with min distance
        min_idx = np.argmin(dists)
        i = int(min_idx / (n_nodes * 2))
        j = min_idx % (n_nodes * 2)

        # merge communities i and j
        ignore.update([i, j])
        merges.append((i, j))
        com_idx = n_nodes + len(merges)
        idx_to_com[com_idx] = idx_to_com[i] + idx_to_com[j]

        # fill dists at com_idx
        for idx, c in idx_to_com.items():
            if idx != com_idx and idx not in ignore:
                dist = _walktrap_com_var(c, idx_to_com[com_idx], probs, t)
                dists[idx, com_idx] = dist
                dists[com_idx, idx] = dist

        # remove merged elements from distance matrix
        fill = np.full((n_nodes * 2,), np.inf)
        dists[i, :] = fill
        dists[:, i] = fill
        dists[j, :] = fill
        dists[:, j] = fill

        # replace coms
        coms.remove(idx_to_com[i])
        coms.remove(idx_to_com[j])
        coms.add(idx_to_com[com_idx])
        for n in idx_to_com[com_idx]:
            idx_to_com[n] = idx_to_com[com_idx]

        # calculate modularity of merge
        memberships.append(_build_membership_list(idx_to_com, n_nodes))
        modularities.append(g.modularity(memberships[-1]))

    optimal_idx = modularities.index(max(modularities))
    if n_clusters != None:
        optimal_idx = len(modularities) - n_clusters
    optimal_count = len(modularities) - optimal_idx

    result = VertexDendrogram(g, merges, optimal_count=optimal_count)
    return result, VertexClustering(g, membership=memberships[optimal_idx])


def walktrap_cf(g, t=4, n_clusters=None):
    # build transition matrix
    n_nodes = len(g.vs)
    degree = np.zeros(shape=(n_nodes, n_nodes))
    adjacent = np.zeros(shape=(n_nodes, n_nodes))
    for v in g.vs:
        for n in v.neighbors():
            adjacent[v.index][n.index] = 1
        degree[v.index][v.index] = v.degree()

    transition = np.linalg.inv(degree) @ adjacent
    probs = fractional_matrix_power(degree, -0.5) @ transition

    dist_f = lambda i, j: _walktrap_com_var([i], [j], probs, t) if i != j else np.inf
    
    return _walktrap_solver(g, probs, t, n_clusters, dist_f)

def _walktrap_random_walk(g, s, t):
    n_nodes = len(g.vs)
    start_node = g.vs[s]
    curr_node = start_node
    while t > 0:
        neighbors = curr_node.neighbors()
        curr_node = neighbors[np.random.randint(len(neighbors))]
        t -= 1

    return start_node.index, curr_node.index

def _walktrap_run_k_walks(g, t, k):
    n_nodes = len(g.vs)
    result = np.full((n_nodes, n_nodes), 0)
    for i in range(n_nodes):
        for _ in range(k):
            start, end = _walktrap_random_walk(g, i, t)
            result[start, end] += 1

    return result / k

def walktrap_sim(g, t=4, k=100, n_clusters=None):
    probs = _walktrap_run_k_walks(g, t, k)

    dist_f = lambda i, j: _walktrap_com_var([i], [j], probs) if i != j else np.inf

    return _walktrap_solver(g, probs, t, n_clusters, dist_f)

