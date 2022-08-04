import numpy as np
import scipy.sparse as sp
from itertools import combinations

def transform(pairs, v_threshold=30, e_threshold=30):
    """construct line expansion from original hypergraph
    INPUT:
        - pairs <matrix>
            - size: N x 2. N means the total vertex-hyperedge pair of the hypergraph
            - each row contains the idx_of_vertex, idx_of_hyperedge
        - v_threshold: vertex-similar neighbor sample threshold
        - e_threshold: hyperedge-similar neighbor sample threshold
    Concept:
        - vertex, hyperedge: for the hypergraph
        - node, edge: for the induced simple graph
    OUTPUT:
        - adj <sparse coo_matrix>: N_node x N_node
        - Pv <sparse coo_matrix>: N_node x N_vertex
        - PvT <sparse coo_matrix>: N_vertex x N_node
        - Pe <sparse coo_matrix>: N_node x N_hyperedge
        - PeT <sparse coo_matrix>: N_hyperedge x N_node
    """
    
    # get # of vertices and encode them starting from 0
    uniq_vertex = np.unique(pairs[:, 0])
    N_vertex = len(uniq_vertex)
    pairs[:, 0] = list(map({vertex: i for i, vertex in enumerate(uniq_vertex)}.get, pairs[:, 0]))
    
    # get # of hyperedges and encode them starting from 0
    uniq_hyperedge = np.unique(pairs[:, 1])
    N_hyperedge = len(uniq_hyperedge)
    pairs[:, 1] = list(map({hyperedge: i for i, hyperedge in enumerate(uniq_hyperedge)}.get, pairs[:, 1]))
    
    N_node = pairs.shape[0]

    # vertex projection: from vertex to node
    Pv = sp.coo_matrix((np.ones(N_node), (np.arange(N_node), pairs[:, 0])), 
        shape=(N_node, N_vertex), dtype=np.float32) # (N_node, N_vertex)
    # vertex back projection (Pv Transpose): from node to vertex
    
    weight = np.ones(N_node)
    for vertex in range(N_vertex):
        tmp = np.where(pairs[:, 0] == vertex)[0]
        weight[tmp] = 1. / len(tmp)
    PvT = sp.coo_matrix((weight, (pairs[:, 0], np.arange(N_node))), 
        shape=(N_vertex, N_node), dtype=np.float32) # (N_vertex, N_node)
    
    # hyperedge projection: from hyperedge to node
    Pe = sp.coo_matrix((np.ones(N_node), (np.arange(N_node), pairs[:, 1])), 
        shape=(N_node, N_hyperedge), dtype=np.float32) # (N_node, N_hyperedge)
    # hyperedge back projection (Pe Transpose): from node to hyperedge
    weight = np.ones(N_node)
    for hyperedge in range(N_hyperedge):
        tmp = np.where(pairs[:, 1] == hyperedge)[0]
        weight[tmp] = 1. / len(tmp)
    PeT = sp.coo_matrix((weight, (pairs[:, 1], np.arange(N_node))), 
        shape=(N_hyperedge, N_node), dtype=np.float32) # (N_node, N_hyperedge)
        
    # construct adj
    edges = []
    # get vertex-similar edges
    for vertex in range(N_vertex):
        position = np.where(pairs[:, 0]==vertex)[0]
        if len(position) > v_threshold:
            position = np.random.choice(position, v_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges += list(tmp_edge)
        else:
            edges += list(combinations(position, r=2))
    
    # get hyperedge-similar edges
    for hyperedge in range(N_hyperedge):
        position = np.where(pairs[:, 1]==hyperedge)[0]
        if len(position) > e_threshold:
            position = np.random.choice(position, e_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges += list(list(tmp_edge))
        else:
            edges += list(combinations(position, r=2))

    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N_node, N_node), dtype=np.float32)
    
    return adj, Pv, PvT, Pe, PeT
