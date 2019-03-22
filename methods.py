import numpy as np
from scipy.linalg import fractional_matrix_power
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

# def modularity(g):

# def _walktrap_dist(a, b, probs):

def walktrap_cf(g):
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

    # initialize n communities with 1 element
    comms = [[i] for i in range(len(g.vs))]

    # iteratively merge communities that are closest, recompute distance

# def walktrap_sim(g):

