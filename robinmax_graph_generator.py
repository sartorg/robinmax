""" Graph generator.

This module contains the functions to generate some
particular types of random graphs.

Thanks to Francis Song for the Watts-Strogatz model:
http://www.nervouscomputer.com/hfs/super-simple-watts-strogatz/

"""

import numpy as np
from scipy.linalg import circulant
import robinmax_graph as gr

def sw_graph_generator(num_nodes, average_degree, rewiring_prob,
                       num_graphs, random_seed):

    rnd = np.random.RandomState(random_seed)

    for i_graph in range(num_graphs):
        p0 = average_degree/(num_nodes-1)
        # A is the adjecency matrix
        A = watts_strogatz(num_nodes, p0, rewiring_prob, directed=True,
                        rngseed=random_seed)

        # Matrix of influence values (weights)
        W = rnd.randint(1, 11, (num_nodes, num_nodes), dtype=int)
        W = np.multiply(W, A)

        # Computing threhsolds
        sum_W = np.sum(W, axis=0)
        sum_A = np.sum(A, axis=0)
        mean = 0.5 * sum_W
        std = np.sqrt(sum_W/sum_A)
        z = rnd.randn(num_nodes) * std + mean
        threhsolds = np.round(np.maximum(np.ones(num_nodes), 
                                        np.minimum(z, sum_W)))
        
        arcs = list()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if A[i][j] > 0.5:
                    arcs.append((i+1, j+1, W[i][j]))
        
        # Create graph
        G = gr.LinearThresholdGraph(threhsolds, arcs)

        # Filename
        filename = "SW-n{:d}-k{:d}-b{:2.1f}-d1-10-g0.5-i{:d}.txt".format(num_nodes,
                                    average_degree, rewiring_prob, i_graph + 1)
        
        # Save graph
        gr.write_text_graph(G, filename)
 
def _distance_matrix(L):
    Dmax = L//2

    D  = list(range(Dmax+1))
    D += D[-2+(L%2):0:-1]
 
    return circulant(D)/Dmax
 
def _pd(d, p0, beta):
    return beta*p0 + (d <= p0)*(1-beta)
 
def watts_strogatz(L, p0, beta, directed=False, rngseed=1):
    """
    Watts-Strogatz model of a small-world network
 
    This generates the full adjacency matrix, which is not a good way to store
    things if the network is sparse.
 
    Parameters
    ----------
    L        : int
               Number of nodes.
 
    p0       : float
               Edge density. If K is the average degree then p0 = K/(L-1).
               For directed networks "degree" means out- or in-degree.
 
    beta     : float
               "Rewiring probability."
 
    directed : bool
               Whether the network is directed or undirected.
 
    rngseed  : int
               Seed for the random number generator.
 
    Returns
    -------
    A        : (L, L) array
               Adjacency matrix of a WS (potentially) small-world network.
 
    """
    rng = np.random.RandomState(rngseed)
 
    d = _distance_matrix(L)
    p = _pd(d, p0, beta)
 
    if directed:
        A = 1*(rng.random_sample(p.shape) < p)
        np.fill_diagonal(A, 0)
    else:
        upper = np.triu_indices(L, 1)
 
        A          = np.zeros_like(p, dtype=int)
        A[upper]   = 1*(rng.rand(len(upper[0])) < p[upper])
        A.T[upper] = A[upper]
 
    return A

if (__name__ == '__main__'):

    #sw_graph_generator(250, 16, 0.1, 1, 1)
    #exit()

    betas = [0.1, 0.3]
    nodes = [1000]
    degrees = [8, 12, 16]
    seed = 1
    for beta in betas:
        for node in nodes:
            for degree in degrees:
                sw_graph_generator(node, degree, beta, 5, seed)
                seed += 1