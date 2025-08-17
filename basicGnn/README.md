# Graph Convolutional Networks (GCNs) on Cora

This document explains how the provided PyTorch Geometric code works for node classification on the **Cora citation dataset** using a **Graph Convolutional Network (GCN)**.

---

## 1. The Dataset (Cora)

- **Nodes = papers** (2708 papers).
- **Edges = citations** (5429 edges, treated as undirected).
- **Node features = bag-of-words** (1433-dimensional feature vector for each paper).
- **Labels = research topics** (7 classes, e.g., Neural Networks, Reinforcement Learning, etc.).

**Goal**: Predict each paper’s topic using graph structure + content.

---

## 2. Graph Convolutional Networks (GCNs)

A GCN updates each node’s representation by **aggregating neighbor features**.

The update rule is:

![GCN Update Rule](https://latex.codecogs.com/png.latex?%5Cbg_white%20H%5E%7B(l%2B1)%7D%20%3D%20%5Csigma%5Cbig(%20%5Ctilde%7BD%7D%5E%7B-1/2%7D%20%5Ctilde%7BA%7D%20%5Ctilde%7BD%7D%5E%7B-1/2%7D%20H%5E%7B(l)%7D%20W%5E%7B(l)%7D%5Cbig))

Where:
- ![A](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctilde%7BA%7D%20%3D%20A%2BI) = adjacency matrix with self-loops.
- ![D](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctilde%7BD%7D) = degree matrix.
- ![H^(l)](https://latex.codecogs.com/png.latex?%5Cbg_white%20H%5E%7B(l)%7D) = node features at layer *l*.
- ![W^(l)](https://latex.codecogs.com/png.latex?%5Cbg_white%20W%5E%7B(l)%7D) = learnable weights.
- ![σ](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Csigma) = nonlinearity (ReLU).

---

## 3. Model Architecture

Your code defines a **2-layer GCN**:

```python
self.conv1 = GCNConv(dataset.num_node_features, 16)
self.conv2 = GCNConv(16, dataset.num_classes)
