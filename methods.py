import numpy as np

def spectral_bisection(g):
    n_nodes = len(g.vs)
    laplacian = np.zeros(shape=(n_nodes, n_nodes))
    print(g.vs[0].neighbors()[0].index)
    for v in g.vs:
        laplacian[v.index][v.index] = v.degree()
        for n in v.neighbors():
            laplacian[v.index][n.index] = -1

    print(laplacian)
    for i in range(min(5, len(laplacian))):
        print(i, sum(laplacian[:,i]), sum(laplacian[i,:]))
    
# def edge_betweenness(g):

# def modularity(g):

# def walktrap(g):

