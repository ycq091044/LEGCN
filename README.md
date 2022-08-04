# Hypergraph Learning with Line Expansion

A very interesting paper finished on Feb. 2020 when I was in my previous group. Unfortunately, I have been so reluctant and cannot find a chance to invest time for further improvements. Recently, I polished it a bit and get it into CIKM'22 Long Research Paper Track (lucky!). **This work proposes an elegant hypergraph transformation (i.e., line expansion), which bijectively maps a hypergraph to a simple graph and enables all existing graph learning algorithms to work effortlessly on hypergraphs.**

> **Hypergraphs** are generalized graphs, which consist of vertices and hyperedges. One vertex in hypergraph can connect to multiple hyperedges and one hyperedge can connect to multiple vertices. Simple graphs are special 2-regular hypergraphs (since each edge only connect to two nodes). 

## 1. Code Structure
- ```data/```: under this folder, we provide several hypergraph datasets
    - 20newsW100, ModelNet40, Mushroom, NTU2012, zoo
- ```src/```
    - **layers.py**: standard neural network layers
    - **models.py**: GCN, GAT, SpGAT (sparse GAT model)
    - **utils.py**: auxiliary functions
    - **main.py**: the running script
    - **LE.py**: the LE transformation script (from hypergraphs to graphs)
- ```config/```: containing the hyperparameter configurations

## 2. Quick Start
### 2.1 Work with Our Hypergraphs
```python
# select a dataset from 20newsW100, ModelNet40, Mushroom, NTU2012, zoo
python main.py --dataset [DATASET] --hidden 64 --dropout 0.5 --lr 0.02 --epochs 50
```
### 2.2 Work with Your Own Hypergraphs
- step 1: create a dataset folder under ```data/DATASET-NAME```
- step 2: process your own hypergraphs into two files: 
    - ```DATASET-NAME.content```: each row is a data sample, starting by sample index, sample features (column based) and sample labels. For example, the first row of **zoo.content**:
        - 0	1	0	0	1	0	0	1	1	1	1	0	0	4	0	0	1	1
        - the first "0" means the sample index
        - the last "1" means label class is 1
        - other float-valued or categorical number in the middle are the features
    - ```DATASET-NAME.edges```: each row is a (vertex index, hyperedge index) pair, the indices do not necessarily start from "0". In our code, we will reindex the vertices and hyperedges. For example, the first row of **zoo.edges**:
        - 2	101
        - "2" means the index for the vertex
        - "101" means the index for the hyperedge
## 3. Plug LE into Your Model?
### step 1: transform the hypergraph into graph
- ```from src.LE import transform```
- Just read the input and output instruction. Prepare your ```DATASET-NAME.edges``` and feed in.
    ```python
    def transform(edges, v_threshold=30, e_threshold=30):
        """construct line expansion from original hypergraph
        INPUT:
            - edges <matrix>
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
    ```
### step 2: run your graph algorithm
```python
# prepare your ```DATASET-NAME.content``` and get features as well as the labels
features, labels = content[:, 1:-1], content[:, -1]

# project features into LE domain
features = Pv @ features

# get embedding in LE graph domain
embedding = run_graph_algorithm(adj, features)

# project back to hypergraph domain
embedding_back = PvT @ embedding

# use FC layer to predict label
class_logit = fully_connected_layer(embedding_back)
```
## 4. Citation
If you think this is repo useful, please cite our paper. For question answering, please contact chaoqiy2@illinois.edu.
```bibtex
@article{yang2022hypergraph,
  title={Semi-supervised Hypergraph Node Classification on Hypergraph Line Expansion},
  author={Yang, Chaoqi and Wang, Ruijie and Yao, Shuochao and Abdelzaher, Tarek},
  booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management, {CIKM} 2022},
  year={2022}
}

@article{yang2020hypergraph,
  title={Hypergraph Learning with Line Expansion},
  author={Yang, Chaoqi and Wang, Ruijie and Yao, Shuochao and Abdelzaher, Tarek},
  journal={arXiv preprint arXiv:2005.04843},
  year={2020}
}
```