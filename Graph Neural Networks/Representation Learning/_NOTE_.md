## Graph Representation Learning

### Symbol

* ${\cal{G}} = ({\cal{V}}, {\cal{E}})$: an undirected graph 
* ${\cal{V}} = \{v_{i}\}_{i=1}^{N}  $：graph nodes set
* $ {\cal{E}} \subseteq {\cal{V}} \times {\cal{V}} $： graph edge set
* $X \in \mathbb{R}^{N \times F}$: graph input feature matrix
  * $x_{i} \in \mathbb{R}^{F}$：attribute of node $v_{i}$
* $A \in \{0,1\}^{N \times N}$: graph adjacency matrix 
* $f$: GNN encoder with parameter $\theta$
  * input: graph structure + node features
* $Z \in {\mathbb{R}}^{N \times F^{'}}$：node embeddings

### Target

* learn a clean adjacency matrix $A^{*}$
* learn corresponding node embeddings $Z^{*} = f(X, A^{*})$

### Pipeline

#### 1. Structure modeling

* encoding function

* models the optimal **graph structure** $A^{*}$ 

* $A^{*}$ represented in edge weights



#### 2. Message propagation

* propagate node features to the refined neighborhoods
* produce an **graph embedding** $Z^{*} = f(A^{*}, X)$



#### 3. Learning objectives

$${\cal{L}} = {\cal{L}}_{task} (Z^{*}, Y) + \lambda {\cal{L}}_{reg} (A^{*}, A)$$

* $${\cal{L}}_{task}$$: task-spacific objective
  * cross entropy for node classification
  * bayesian personalized ranking loss for link prediction
* ${\cal{L}}_{reg}$: regularize the learned graph structure $A^{*}$ to meet some prior topology constraints (sparsity, low rank property)



### Metric Learning Approaches

#### 1. Definition

​	the weight of an edge between two nodes could be represented as **a distance measure** between two end nodes

#### 2. Computation

##### (1) metric function

​	Metric learning approaches refine the graph structure by learning a **metric function $\phi(\cdot,\cdot)$** of a pair of representations

​	model the edge weights

​	$${\tilde{A}}_{ij} = \phi(z_{i}, z_{j})$$

* $A$: original graph structure
* $\tilde{A}$: learned adjacency matrix
* $z_{i}$: learned embedding of node $v_{i}$ produced by GNN
* ${\tilde{A}}_{ij}$: learned edge weight between node $v_{i}$ and $v_{j}$

##### (2) update function

​	the learned edge weights are combined with the original adjacency matrix using an update function $g(\cdot, \cdot)$

​	combine $A$ and $\tilde{A}$

​	$$A^{*} = g(A, \tilde{A})$$

###### a. interpolation combination

​	combine $A$ and $\tilde{A}$ with a hyperparameter $\alpha$ mediating  the influence of the learned structure

$$ A^{*} = \alpha \tilde{A} + (1 - \alpha) A$$

###### b. attentive combination

* employ a channel attention mechanism to fuse $A$ and $\tilde{A}$

* $g$: multilayer perceptrons (MLPs)

##### c. sparsity regularization

​	prune edges according to edge weights as a post-processing operation

* $k$NN graph: each node has up to $k$ neighbors
* $\epsilon$NN graph: edges whose weight are less than $\epsilon$ will be discarded

#### 3. kernel-based approaches

​	employ traditional kernel functions as the metric $\phi$ to model edge weights between two nodes

#### 4. attention-based approaches

​	utilize an attention network or other neural netwoks to capture the interaction among nodes



### Probabilistic Modeling Approaches

#### 1. Definition

##### (1) assumption

* graph is generated via a **sampling process** from certain distributions
* model the probability of sampling edges with learnable parameters

##### (2) target

* make sampling operation differentiable
* enabling optimizing with conventional gradient descent methods



### Direct Optimization Approaches

#### 1. Definition

* treat the graph adjacency matrix $A^{*}$ as learnable parameters
* $A^{*}$ are optimized directly along with GNN parameters $\theta$