import numpy as np
import scipy.sparse as sp
import torch
from LE import transform
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
    id_train = np.random.choice(np.arange(test), train, replace=False)
    id_val = np.random.choice(np.array(list(set(np.arange(test)) - set(id_train))), val, replace=False)
    id_test = np.array(list(set(np.arange(test)) - set(id_train) - set(id_val)))
    if dataset == 'zoo':
        return id_train, id_test, id_test
    return id_train, id_val, id_test

def load_data(path="../data/", dataset="zoo"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path+dataset+'/', dataset),
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    print ('loaded features and labels')
    
    # build graph
    pairs = np.genfromtxt("{}{}.edges".format(path+dataset+'/', dataset),
                                    dtype=np.int32)
    print ('loaded edge pairs')
    
    # transform into LE 
    adj, Pv, PvT, Pe, PeT = transform(pairs)
    print ('get LE adjacency and projections')

    # get dataset split
    idx_train, idx_val, idx_test = catchSplit(dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))
    labels = torch.LongTensor(np.where(labels)[1])
    
    # project features to LE
    features = torch.FloatTensor(np.array(Pv @ features.todense()))
    
    # sparse back projection matrix
    PvT = sparse_mx_to_torch_sparse_tensor(PvT)
    return adj, PvT, features, labels, idx_train, idx_val, idx_test

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
