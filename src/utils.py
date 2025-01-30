import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.nn import Node2Vec
import gc
import sys
import random
from os import path

PREPROCESSPATH = path.dirname(path.abspath(__file__)) + '/preprocessed/'

#######################
def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


#######################
def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#######################
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


#######################
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#######################
def memory_free():
    # Run garbage collection
    gc.collect()

    # Clear unused GPU memory
    torch.cuda.empty_cache()


#######################
def deepwalk_encoding(edge_index, dim, device):

    # reproducibility ensure
    fix_seed()

    # node2vec with p=q=1.0 is equivalent to deepwalk
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    ).to(device)

    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=1024, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    # embedding training
    model.train()
    for _ in range(1, 101):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    # embedding generation
    model.eval()
    with torch.no_grad():
        dw_encode = model().detach().cpu()

    return dw_encode


#######################
def similarity_coarsening(dataset,
                  Omega: torch.Tensor,
                  r: float,
                  device: str,
                  undirected: bool = True):
    """
    Perform graph coarsening on a given graph (A, X) with augmented features Omega.

    Args:
        dataset: The built dataset class. See dataset.py for details.
        Omega (torch.Tensor): Augmented node feature matrix of shape [N, D].
                              Used to compute distances between clusters.
        r (float): Coarsening ratio (0 < r < 1). The process stops when the number
                   of clusters / N <= r.
        undirected (bool): If True, the graph is assumed undirected and the adjacency
                           matrix is made symmetric.

    Returns:
        A_prime (torch.Tensor): Coarsened adjacency matrix of shape [K, K],
                                where K is the final number of clusters.
        X_prime (torch.Tensor): Coarsened feature matrix of shape [K, F].
        partition (list): A list of sets, where each set is the node indices belonging
                          to that cluster in the final partition.
    """

    x = dataset.graph['node_feat'].to(device)
    edge_index = dataset.graph['edge_index']

    N = x.size(0)

    # ------------------------------------------------------------------
    # Build dense adjacency A from edge_index with assuming an unweighted graph.
    # ------------------------------------------------------------------
    A = torch.zeros((N, N), dtype=x.dtype, device=device)
    rows = edge_index[0]
    cols = edge_index[1]

    A[rows, cols] = 1.0
    if undirected:
        A[cols, rows] = 1.0

    del edge_index, rows, cols

    # ------------------------------------------------------------------
    # Initialize each node in its own cluster: P = { {0}, {1}, ..., {N-1} }
    # ------------------------------------------------------------------
    partition = [{i} for i in range(N)]

    # ------------------------------------------------------------------
    # Helper function to get cluster centroids from partition
    # ------------------------------------------------------------------
    def get_cluster_centroids(part):
        """
        part: list of sets, each set is a cluster of node indices
        returns: a tensor of shape [num_clusters, D]
                 where row i is the average of Omega over cluster i
        """
        centroids = []
        for cluster in part:
            cluster_indices = torch.tensor(list(cluster), device=device)
            # Average augmented features
            cluster_omega = Omega[cluster_indices]  # shape [size_of_cluster, D]
            centroid = cluster_omega.mean(dim=0)    # shape [D]
            centroids.append(centroid)
        return torch.stack(centroids, dim=0)  # shape [k, D]

    # ------------------------------------------------------------------
    # Repeatedly merge the two closest clusters (by Euclidean distance
    #    of their Omega centroids) until (#clusters / N) <= r.
    # ------------------------------------------------------------------
    while len(partition) / N > r:
        # Compute cluster centroids
        centroids = get_cluster_centroids(partition)  # shape [k, D]
        k = centroids.size(0)

        # Compute pairwise distances (k x k)
        dist_matrix = torch.cdist(centroids, centroids, p=2)
        # To avoid picking the diagonal, set it to large value
        dist_matrix.fill_diagonal_(float('inf'))

        # Find the minimum distance in the upper (or entire) triangle
        min_val = dist_matrix.view(-1).min(dim=0)
        min_idx = min_val.indices  # index in the flattened distance matrix

        # unravel the min_idx into (i, j)
        i = min_idx // k
        j = min_idx % k
        if i > j:
            i, j = j, i  # ensure i < j for safe pop

        # Merge cluster j into cluster i, remove cluster j
        partition[i] = partition[i].union(partition[j])
        partition.pop(j)

    # Now we have our final partition
    final_num_clusters = len(partition)

    # ------------------------------------------------------------------
    # Build the cluster membership matrix P of shape [K, N].
    #    P[i, n] = 1 if node n is in cluster i, otherwise 0.
    # ------------------------------------------------------------------
    P = torch.zeros((final_num_clusters, N), dtype=x.dtype, device=device)
    cluster_sizes = []
    for i, cluster in enumerate(partition):
        cluster_list = list(cluster)
        P[i, cluster_list] = 1.0
        cluster_sizes.append(len(cluster))

    cluster_sizes = torch.tensor(cluster_sizes, dtype=x.dtype, device=device)  # shape [K]
    cluster_sizes = cluster_sizes.unsqueeze(-1)  # shape [K, 1], for broadcasting

    # ------------------------------------------------------------------
    # Compute coarsened adjacency A' = P @ A @ P^T
    #    This will be of shape [K, K].
    # ------------------------------------------------------------------
    A_prime = P @ A @ P.t()
    edge_index_prime = A_prime.nonzero(as_tuple=False).T  # Shape: (2, num_edges)

    del A_prime, A
    memory_free()

    # ------------------------------------------------------------------
    # Compute coarsened features X' = C^-1 * P * X
    #    where C is diag(cluster_sizes). For each cluster i,
    #    we divide by the cluster size to get the average feature.
    # ------------------------------------------------------------------
    x_prime = P @ x  # shape [K, F]
    x_prime = x_prime / cluster_sizes  # broadcast division by each cluster size

    return ( edge_index_prime, x_prime, (P / cluster_sizes).t().to_sparse() )