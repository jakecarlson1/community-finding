import numpy as np

def spectral_bisection(g):
    n_nodes = len(g.vs)
    laplacian = np.zeros(shape=(n_nodes, n_nodes))
    for v in g.vs:
        for n in v.neighbors():
            laplacian[v.index][n.index] = -1
        laplacian[v.index][v.index] = v.degree()

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
    print(s1 - s2)
    print(s2 - s1)

    # find min vertex cover of vertices
    
# def edge_betweenness(g):

# def modularity(g):

# def walktrap(g):

