import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor
from models import GCN, SpGAT, GAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=200, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--fastmode', type=int, default=0,
                    help='Validate during training pass.')
parser.add_argument('--weight_l2', type=float, default=1.5e-3,
                    help='weight for parameter L2 regularization')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--modelType', type=int, default=0,
                    help='GCN (0), SpGAT (1), GAT (2)')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Name of dataset')

args = parser.parse_args()

def train(model, epoch):
    tic = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, PvT)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    l2 = 0
    for p in model.parameters():
        l2 = l2 + (p ** 2).sum()
    loss_train = loss_train + args.weight_l2 * l2

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj, PvT)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - tic), flush=True)

def test(model):
    model.eval()
    output = model(features, adj, PvT)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, PvT, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)
    
# model definition
if args.modelType == 0:
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)   
elif args.modelType == 1:
    adj = torch.FloatTensor(np.array(adj.todense()))
    model = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
elif args.modelType == 2:
    adj = torch.FloatTensor(np.array(adj.todense()))
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    
if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    Pvp = PvT.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    model.cuda()
    
    
optimizer = optim.Adam(model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)

# train model
tic = time.time()
for epoch in range(args.epochs):
    train(model, epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - tic))

#test model
test(model)
