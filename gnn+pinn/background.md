# GNNs & PINNs for 1D Heat Diffusion: A Primer

This document outlines the fundamental concepts of combining a **Graph Neural Network (GNN)** with a **Physics-Informed Neural Network (PINN)** to model a simple physical system: heat diffusion along a one-dimensional rod.

---

## 1. The System as a Graph (The GNN Component)

First, we represent the physical system as a graph. For a 1D rod, this is straightforward:

- **Nodes**: The rod is divided into a finite number of segments. Each segment is a node in the graph. The primary feature of each node *i* is its temperature, ![u_i](https://latex.codecogs.com/svg.latex?u_i).
- **Edges**: Each node is connected to its immediate neighbors by an edge. This represents the path through which heat can flow between adjacent segments.

The objective of the GNN is to learn a function that predicts the temperature of all nodes at the next time step, ![t+Δt](https://latex.codecogs.com/svg.latex?t%2B%5CDelta%20t), given the temperatures at the current time step, ![t](https://latex.codecogs.com/svg.latex?t).

![u_pred(t+Δt)=GNN(u(t),A)](https://latex.codecogs.com/svg.latex?u_%7Bpred%7D(t%2B%5CDelta%20t)%20%3D%20GNN(u(t)%2C%20A))

Where:
- ![u(t)](https://latex.codecogs.com/svg.latex?u(t)) is the vector of all node temperatures at time ![t](https://latex.codecogs.com/svg.latex?t).
- ![A](https://latex.codecogs.com/svg.latex?A) is the adjacency matrix representing the graph's structure.

---

## 2. The Governing Physics (The PINN Component)

The physical behavior of heat flow is described by the **1D Heat Equation**, a partial differential equation (PDE):

![∂u/∂t = α ∂²u/∂x²](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20%3D%20%5Calpha%20%5Cfrac%7B%5Cpartial%5E2%20u%7D%7B%5Cpartial%20x%5E2%7D)

Where:
- ![u(x,t)](https://latex.codecogs.com/svg.latex?u(x%2Ct)) is the temperature at position ![x](https://latex.codecogs.com/svg.latex?x) and time ![t](https://latex.codecogs.com/svg.latex?t).
- ![∂u/∂t](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D) is the rate of change of temperature over time.
- ![∂²u/∂x²](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%5E2%20u%7D%7B%5Cpartial%20x%5E2%7D) (the spatial Laplacian) is the curvature of the temperature profile, which drives diffusion.
- ![α](https://latex.codecogs.com/svg.latex?%5Calpha) is the thermal diffusivity constant of the material.

---

## 3. The Hybrid Loss Function: Fusing Data and Physics

The core of the PINN approach is to train the GNN using a composite loss function that includes terms for both **data** and **physics**.

![L_total = L_data + λ L_physics](https://latex.codecogs.com/svg.latex?L_%7Btotal%7D%20%3D%20L_%7Bdata%7D%20%2B%20%5Clambda%20L_%7Bphysics%7D)

### Data Loss (L_data)

This is the standard supervised learning loss. If we have sensor readings for a few specific nodes, we penalize the model for the difference between its prediction and the true, measured temperature.

For a set of known data points ![S](https://latex.codecogs.com/svg.latex?S), we use a metric like Mean Squared Error (MSE):

![L_data = (1/|S|) Σ (u_pred,i(t+Δt) - u_actual,i(t+Δt))²](https://latex.codecogs.com/svg.latex?L_%7Bdata%7D%20%3D%20%5Cfrac%7B1%7D%7B%7CS%7C%7D%20%5Csum_%7Bi%20%5Cin%20S%7D%20(u_%7Bpred%2Ci%7D(t%2B%5CDelta%20t)%20-%20u_%7Bactual%2Ci%7D(t%2B%5CDelta%20t))%5E2)

This loss "anchors" the simulation to reality, but it's often sparse.

### Physics Loss (L_physics)

This is the innovative part. We enforce the heat equation on all nodes in the graph, even those without sensor data. We do this by calculating the **residual of the PDE** — how much the model's output "disobeys" the physical law.

**Discretization:**

- **Time Derivative:** Approximated using the GNN's input and output.

  ![∂u/∂t ≈ (u_pred(t+Δt) - u(t)) / Δt](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20%5Capprox%20%5Cfrac%7Bu_%7Bpred%7D(t%2B%5CDelta%20t)%20-%20u(t)%7D%7B%5CDelta%20t%7D)

- **Spatial Laplacian:** Approximated on the graph. For a node *i*, it's the difference between its neighbors' temperatures and its own.

  ![∇²u_i(t) = u_(i-1)(t) + u_(i+1)(t) - 2u_i(t)](https://latex.codecogs.com/svg.latex?%5Cnabla%5E2%20u_i(t)%20%3D%20u_%7Bi-1%7D(t)%20%2B%20u_%7Bi%2B1%7D(t)%20-%202u_i(t))

The physics loss is the mean squared error of the PDE's residual across all nodes ![N](https://latex.codecogs.com/svg.latex?N):

![L_physics = (1/|N|) Σ ( ( (u_pred,i(t+Δt) - u_i(t)) / Δt - α ∇²u_i(t) )² )](https://latex.codecogs.com/svg.latex?L_%7Bphysics%7D%20%3D%20%5Cfrac%7B1%7D%7B%7CN%7C%7D%20%5Csum_%7Bi%20%5Cin%20N%7D%20%5CBigg(%20%5Cfrac%7Bu_%7Bpred%2Ci%7D(t%2B%5CDelta%20t)%20-%20u_i(t)%7D%7B%5CDelta%20t%7D%20-%20%5Calpha%20%5Cnabla%5E2%20u_i(t)%20%5CBigg)%5E2)

By minimizing this loss, we force the GNN to learn the underlying physics of diffusion. The hyperparameter ![λ](https://latex.codecogs.com/svg.latex?%5Clambda) balances the importance of fitting the sparse data versus obeying the physical law.
