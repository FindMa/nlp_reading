## Self Supervised Learning on Graph

### Symbols

* $G = (V, E, \alpha)$: an attribute undirected graph 
* $V = \{v_{1}, ..., v_{|V|} \}  $：graph nodes set
* $ E = \{e_{1}, ..., e_{|E|}\}$: graph edge set
* $\alpha: V \rightarrow \mathbb{R}^{d}$: mapping from a node to corresponding attribute
* $A \in \mathbb{R}^{|V| \times |V|}$: graph adjacency matrix
* $X \in \mathbb{R}^{|V| \times d}$: graph input feature matrix

* $f$: graph encoder
* $H = f(A, X)$: graph encoder output feature
  * node-level encoder:
    *  $f_{n}: \mathbb{R}^{|V|\times|V|} \times \mathbb{R}^{|V|\times d} \rightarrow \mathbb{R}^{|V| \times q}$
    * $H_{node}=f_{node}(A, X) \in \mathbb{R}^{|V| \times q}$
  * graph-level encoder: 
    *  $f_{g}: \mathbb{R}^{|V|\times|V|} \times \mathbb{R}^{|V|\times d} \rightarrow \mathbb{R}^{q}$
    * $h_{graph}=f_{graph}(A, X) \in \mathbb{R}^{q}$



### Self-Supervised Learning

```
The key idea of these methods is to define pretext training tasks to capture and use the dependencies among different dimensions of the input data
```

#### 1. Contrastive Model

* utilize **self-supervision** to learn data representation
* perform **pre-training** for downstream tasks
* perform **discrimination** between positive pairs and negative pairs

```
To obtain good representations of graphs and perform effective pre-training, self-supervised models are supposed to capture essential information from both nodes attributes and structural topology of graphs.
the key challenge lies in how to obtain good views of graphs and the selection of graph encoder for different models and datasets
graph-level encoders are usually onstructed as a node-level encoder followed by a readout function
Given training graphs, contrastive learning aims to learn one or more encoders such that representations of similar graph instances agree with each other, and that representations of dissimilar graph instances disagree with each other.
```

* **node-level feature**: local feature

* **graph-level feature**: global feature



#### 2. Predictive Model

* trained in a **supervised** fashion
* **labels** are generated based on certain properties
  of the input data or by selecting certain parts of the data
* consist of an **encoder** and one or more **prediction heads**



### Contrastive Learning Framework

#### 1. Pipeline

##### (1) Process

* **Transformation**: transformations that compute multiple views from each given graph
* **Graph Encoder**: encoders that compute the representation for each view
* **Contrastive Objectives**: the learning objective to optimize parameters in encoders

##### (2) Symbol

* $G = (A,X)$：graph data, treated as a random variable
* $\cal{T}_{1}, ..., \cal{T}_{k}$：multiple transformations applied to obtain different views
* $w_{1}, ..., w_{k}$: different graph views of the graph
* $f_{1}, ..., f_{k}$：graph encoder network
* $h_{1}, ..., h_{k}$：graph encoder output feature representations

##### (3) Computation

$$w_{i} = {\cal{T}}_{i} (A, X) = (\hat{A}_{i}, \hat{X}_{i})$$

$$h_{i} = f_{i}(w_{i})$$

##### (4) Contrastive objective

$$ \mathop{max}\limits_{\{f_{i}\}_{i=1}^{k}} \frac{1}{\sum_{i \neq j} \sigma_{ij}} [\mathop{\sum}\limits_{i \neq j} \sigma_{ij} {\cal{I}} (h_{i},h_{j})]$$

$\cal{I}(h_{i}, h_{j})$ is the mutual information(互信息) between a pair of feature representations $h_{i}$ and $h_{j}$

#### 2. Graph View Transformation

##### (1) Feature-space Transformation

* perform the transformation on the feature matrix $X$

$${\cal{T}}_{X}: \mathbb{R}^{|V| \times d} \rightarrow \mathbb{R}^{|V| \times d} $$

$${\cal{T}_{feature}}(A,X) = (A, {\cal{T}}_{X}(X))$$

###### a. Node attribute masking

$${\cal{T}}_{X}^{(mask)} (X) = X * (1-{\mathbb{1}}_{m}) + M * {\mathbb{1}}_{m}$$



##### (2) Structure-space Transformation

* perform the transformation on the adjace matrix $A$

$${\cal{T}}_{A}: \mathbb{R}^{|V| \times |V|} \rightarrow \mathbb{R}^{|V| \times |V|} $$

$${\cal{T}_{structure}}(A,X) = ({\cal{T}}_{A}(A), X)$$

###### a. Edge perturbation

$$ {\cal{T}}_{A}^{(pert)} (A) = A * (1 - {\mathbb{1}}_{p}) + (1 - A) * {\mathbb{1}}_{p}$$

###### b. Graph diffusion

$$ {\cal{T}}_{A}^{(heat)} (A) = exp(tAD^{-1} - t)$$

$$ {\cal{T}}_{A}^{(PPR)} (A) = \alpha(I_{n} - (1 - \alpha) D^{-1/2} A D^{-1/2})^{-1}$$



##### (3) Sample-based Transformation

$${\cal{T}}_{sample} (A, X) = (A[S;S], X[S])$$

* $S \subseteq V$： denote a subset of nodes
* $[\cdot]$：denote select certain rows or columns from a matrix based on indices or nodes in $S$

###### a. Uniform sampling

###### b. Ego-nets sampling

###### c. Random walk

#### 3. Graph Encoder

##### (1) computation

$$x_{v}^{(k)} = COMBINE^{(k)} (x_{v}^{(k-1)}, a_{v}^{k})$$

$$a_{v}^{k} = AGGREGATE^{(k)} ( (x_{v}^{(k-1)}, x_{u}^{(k-1)}): u \in {\cal{N}} (v) ) $$



