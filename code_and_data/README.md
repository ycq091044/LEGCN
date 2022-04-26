# Code Instructions
data and code for AAAI2021 submission "Hypergraph Representation Learning with Line Expansion"

### how to run the codes
Uage: train.py [-h] [--no-cuda] [--seed SEED] [--epochs EPOCHS] [--lr LR]
                [--fastmode FASTMODE] [--weight_l2 WEIGHT_L2]
                [--weight_decay WEIGHT_DECAY] [--hidden HIDDEN]
                [--dropout DROPOUT] [--modelType MODELTYPE] [--LE LE]
                [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --no-cuda             Disables CUDA training.
  --seed SEED           Random seed.
  --epochs EPOCHS       Number of epochs to train.
  --lr LR               Initial learning rate.
  --fastmode FASTMODE   Validate during training pass.
  --weight_l2 WEIGHT_L2
                        weight for parameter L2 regularization
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 loss on parameters).
  --hidden HIDDEN       Number of hidden units.
  --dropout DROPOUT     Dropout rate (1 - keep probability).
  --modelType MODELTYPE
                        GCN (0), SpGAT (1), GAT (2)
  --LE LE               on graphs (0), on line expansion (1)
  --dataset DATASET     Name of dataset

### default parameters
[cora]
lr = 0.002
hidden = 32
weight_l2 = 1.5e-3

[citeseer]
lr = 0.0005 / 0.001
hidden = 64
weight_l2 = 0.07

[pubmed]
lr = 0.002 / 0.01
hidden = 32
weight_l2 = 1.5e-3

### recommended running setting
python train.py --dataset 20newsW100  --LE 1 --weight_l2 1e-4 --lr 0.001 --dropout 0.5
python train.py --dataset Mushroom --hidden 128 --LE 1 --lr 0.01 --weight_l2 1e-3
python train.py --dataset NTU2012 --LE 1 --lr 0.01 --hidden 128
python train.py --dataset ModelNet40 --lr 0.01 --hidden 128 --LE 1
python train.py --dataset zoo --LE 1 --hidden 32 --dropout 0.5 --lr 0.02