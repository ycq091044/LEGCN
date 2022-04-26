import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from itertools import combinations
import configparser

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def catchSplit(dataset):
    cp = configparser.SafeConfigParser()
    cp.read('../config/LE.conf')
    split = []
    train, val, test = cp[dataset]['id_train'], cp[dataset]['id_val'], cp[dataset]['id_test']
    train, val, test = list(map(int, [train, val, test]))
    if dataset in ['cora', 'citeseer', 'pubmed']:
        id_train, id_val, id_test = range(train), range(train, val), range(val, test)
    elif dataset in ['20newsW100', 'ModelNet40', 'NTU2012', 'Mushroom', 'zoo']:
        id_train = np.random.choice(np.arange(test), train, replace=False)
        id_val = np.random.choice(np.array(list(set(np.arange(test)) - set(id_train))), val, replace=False)
        id_test = np.array(list(set(np.arange(test)) - set(id_train) - set(id_val)))
    if dataset == 'zoo':
        return id_train, id_test, id_test
    return id_train, id_val, id_test

def hyperSetting(dataset):
    cp = configparser.SafeConfigParser()
    cp.read('../config/LE.conf')
    nodeNum, heNum = cp[dataset]['nodeNum'], cp[dataset]['heNum']
    return int(nodeNum), int(heNum)
    
def catchSample(dataset):
    cp = configparser.SafeConfigParser()
    cp.read('../config/LE.conf')
    tempParam = []
    for _, v in cp.items(dataset)[:4]:
        tempParam.append(int(v))
    return tempParam

def fuseWeight(edges, nodeNum, dataset, heNum):
    inverseMap = {}
    for hyperEdge in range(nodeNum, nodeNum + heNum):
        inverseMap[hyperEdge] = 1.0 / len(np.where(edges==hyperEdge))
    inverseE = []
    # each hyperE contains exactly two vertices
    if dataset in ['cora', 'citeseer', 'pubmed']:
        for _ in edges:
            inverseE.append(0.5); inverseE.append(0.5)
    elif dataset in ['20newsW100', 'ModelNet40', 'NTU2012', 'Mushroom', 'zoo']:
        for _, v in edges:
            inverseE.append(inverseMap[v])
    return inverseE

def lineExpansion(edges, nodeNum, dataset, heNum):
    """construct line expansion from original hypergraph"""
    n2he = []
    for he, (u, v) in enumerate(edges):
        if dataset in ['cora', 'citeseer', 'pubmed']:
            n2he.append([he, u]); n2he.append([he, v])
        elif dataset in ['20newsW100', 'ModelNet40', 'NTU2012', 'Mushroom', 'zoo']:
            n2he.append([he, u])
    n2he = np.array(n2he)

    # Vertex Projection Pv
    Pv = sp.coo_matrix((fuseWeight(edges, nodeNum, dataset, heNum), (n2he[:, 0], n2he[:, 1])),
                        shape=(edges.shape[0], nodeNum), dtype=np.float32)
        
    np.random.seed(100)

    newEdge = []
    nt, sn, et, se = catchSample(dataset)
    # construct adj
    for node in range(nodeNum):
        position = np.where((edges==node).sum(axis=1) == 1)[0]
        if len(position) > nt:
            possition = np.random.choice(position, nt, replace=False)
            possibleEdge = np.array(list(combinations(position, r=2)))
            possibleIndex = np.arange(len(possibleEdge))
            # selectIndex = np.random.choice(possibleIndex, nt, replace=False)
            newEdge += list(possibleEdge[possibleIndex])
        else:
            newEdge += list(combinations(position, r=2))
    
    for hyperEdge in range(nodeNum, nodeNum + heNum):
        position = np.where((edges==hyperEdge).sum(axis=1) == 1)[0]
        if len(position) > et:
            position = np.random.choice(position, et, replace=False)
            possibleEdge = np.array(list(combinations(position, r=2)))
            possibleIndex = np.arange(len(possibleEdge))
            # selectIndex = np.random.choice(possibleIndex, se, replace=False)
            newEdge += list(possibleEdge[possibleIndex])
        else:
            newEdge += list(combinations(position, r=2))

    newEdge = np.array(newEdge)

    adj = sp.coo_matrix((np.ones(newEdge.shape[0]), (newEdge[:, 0], newEdge[:, 1])),
                        shape=(edges.shape[0], edges.shape[0]),
                        dtype=np.float32)
    return adj, Pv

def load_data(LE=0, path="../data/", dataset="ModelNet40"):
    """Load dataset: cora, citeseer, pubmed"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path+dataset+'/', dataset),
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    print ('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.edges".format(path+dataset+'/', dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')

    if LE == 0:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
        else: print ('wrong parameter choosing!')
    else:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            adj, Pv = lineExpansion(edges, features.shape[0], dataset, edges.shape[0])
        elif  dataset in ['20newsW100', 'ModelNet40', 'NTU2012', 'Mushroom', 'zoo']:
            nodeNum, heNum = hyperSetting(dataset)
            adj, Pv = lineExpansion(edges, nodeNum, dataset, heNum)
            features = features[:nodeNum, :]
        Pv, Pvp = normalize(Pv), normalize(Pv.T)

    idx_train, idx_val, idx_test = catchSplit(dataset)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if LE == 0:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    else:
        adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))
    labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if LE == 0:
        features = torch.FloatTensor(np.array(features.todense()))
        return adj, features, labels, idx_train, idx_val, idx_test
    else:
        features = torch.FloatTensor(np.array(Pv @ features.todense()))
        Pvp = sparse_mx_to_torch_sparse_tensor(Pvp)
        return adj, Pvp, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = (r_mat_inv).dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
