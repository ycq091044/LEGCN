# Hypergraph Learning with Line Expansion

A very interesting paper finished on Feb. 2020 when I was in my previous group. Unfortunately, I have been so reluctant that I cannot find a chance to invest time for further improvements and re-submission. Until now, I decide to let it go. **This paper utilizes an elegant hypergraph transformation (i.e., line expansion) to enable all existing graph learning algorithms to work effortlessly on hypergraphs.**

### Package Dependency
The package dependency is light: ``pytorch``, ``scipy``, ``numpy``, compatible with recent versions.

### Code Structure
- ```data/```: under this folder, we provide several graph, hypergraph datasets (processed)
    - **Hypergraphs**: 20newsW100, ModelNet40, Mushroom, NTU2012, zoo
    - **graphs**: cora, citeseer, pubmed
- ```src/```
    - **layers.py**: standard neural network layers
    - **models.py**: GCN, GAT, SpGAT (sparse GAT model)
    - **utils.py**: other auxiliary functions
    - **train.py**: the running script
- ```config/```: containing the hyperparameter configurations

### Recommended Setting
```python
"""
usage: train.py [-h] [--no-cuda] [--seed SEED] [--epochs EPOCHS] [--lr LR] [--fastmode FASTMODE]
                [--weight_l2 WEIGHT_L2] [--weight_decay WEIGHT_DECAY] [--hidden HIDDEN] [--dropout DROPOUT]
                [--modelType MODELTYPE] [--LE LE] [--dataset DATASET]
"""
python train.py --dataset 20newsW100  --LE 1 --weight_l2 1e-4 --lr 0.001 --dropout 0.5
python train.py --dataset Mushroom --hidden 128 --LE 1 --lr 0.01 --weight_l2 1e-3
python train.py --dataset NTU2012 --LE 1 --lr 0.01 --hidden 128
python train.py --dataset ModelNet40 --lr 0.01 --hidden 128 --LE 1
python train.py --dataset zoo --LE 1 --hidden 32 --dropout 0.5 --lr 0.02
```

### Citation
If you think this is repo useful, please cite our paper. For question answering, please contact chaoqiy2@illinois.edu.
```bibtex
@article{yang2020hypergraph,
  title={Hypergraph learning with line expansion},
  author={Yang, Chaoqi and Wang, Ruijie and Yao, Shuochao and Abdelzaher, Tarek},
  journal={arXiv preprint arXiv:2005.04843},
  year={2020}
}
```