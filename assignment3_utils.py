import os
import random

import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import pandas as pd
from sklearn import metrics


def nx_list_to_dataloader(nx_graph_list, bs):
    # Create empty lists to store node features, edge indices, and edge features
    data_list = []
    n_nodes_acc = 0

    # Iterate over each NetworkX graph
    for graph in nx_graph_list:
        # Extract node features from the graph
        cur_data = Data(
            x=torch.Tensor(list(graph.nodes())),
            edge_index=torch.Tensor(list(graph.edges())).long().T,
            edge_attr=torch.ones(len(list(graph.edges()))),
            y=torch.tensor(0),
            n_nodes=torch.Tensor([len(list(graph.nodes()))]),
        )
        cur_data.x = torch.ones_like(cur_data.x)
        data_list.append(cur_data)

    return DataLoader(data_list, batch_size=bs, shuffle=True)


def get_nx_dataset(dataset_path):
    nx_dataset = []
    (
        adjs,
        eigvals,
        eigvecs,
        n_nodes,
        max_eigval,
        min_eigval,
        same_sample,
        n_max,
    ) = torch.load(dataset_path)
    for adj in adjs:
        nx_graph = nx.from_numpy_array(adj.detach().cpu().numpy())
        nx_dataset.append(nx_graph)
    num_graphs = len(nx_dataset)
    num_train_graphs = int(0.8 * 0.8 * num_graphs)
    num_val_graphs = int(0.8 * 0.2 * num_graphs)
    train_graphs = nx_dataset[:num_train_graphs]
    val_graphs = nx_dataset[num_train_graphs : num_train_graphs + num_val_graphs]
    test_graphs = nx_dataset[num_train_graphs + num_val_graphs :]
    return {"train": train_graphs, "val": val_graphs, "test": test_graphs}


def get_betas(timesteps=1000, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    x = np.expand_dims(x, 0)  # ((1, steps))

    alphas_cumprod = (
        np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    )  # ((components, steps))
    alphas_cumprod_new = alphas_cumprod / np.expand_dims(alphas_cumprod[:, 0], 1)
    # remove the first element of alphas_cumprod and then multiply every element by the one before it
    alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]

    betas = 1 - alphas  # ((components, steps)) # X, charges, E, y
    betas = np.swapaxes(betas, 0, 1)

    return torch.Tensor(betas)


def get_alphas(timesteps=1000):
    betas = get_betas(timesteps=timesteps)
    alphas = torch.clamp(1 - betas, min=0, max=0.9999)

    return alphas


def get_alphas_bar(alphas):
    log_alphas = torch.log(alphas)
    log_alphas_bar = torch.cumsum(log_alphas, dim=0)
    log_alphas_bar = log_alphas_bar
    alphas_bar = torch.exp(log_alphas_bar)

    return alphas_bar


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)  # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency  # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
    D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack(
            (eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues))
        )
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(
        0
    ) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=5):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat(
            (vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2
        )  # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(
        0
    ) + n_connected.unsqueeze(
        2
    )  # bs, 1, k
    indices = indices.expand(-1, n, -1)  # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """

    def __init__(self, mode):
        """mode: 'eigenvalues' or 'all'"""
        self.mode = mode

    def __call__(self, E_t, mask):
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == "eigenvalues":
            eigvals = torch.linalg.eigvalsh(L)  # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigvals
            )
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == "all":
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigvals
            )

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigvectors, node_mask=mask, n_connected=n_connected_comp
            )
            return (
                n_connected_comp,
                batch_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigenvector,
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """X: bs, n, dx."""
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float(1e5)
    return torch.softmax(x_masked, **kwargs)


def plot_intermediate_graphs(holder_list, T, num_graphs_to_plot):

    sorted_holder_list = sorted(
        holder_list, key=lambda x: -x.y[0]
    )  # we want to see the graphs with time decreasing (noisy -> clean graphs)

    # extract info from placeholder
    num_timesteps = len(holder_list)
    timesteps = []
    nx_graphs_to_plot = []
    for holder in sorted_holder_list:
        nx_graphs_list = holder.to_nx_graph_list()[:num_graphs_to_plot]
        timesteps.append(holder.y[0])
        nx_graphs_to_plot.append(nx_graphs_list)

    # plot
    fig, axes = plt.subplots(num_graphs_to_plot, num_timesteps, figsize=(15, 6))
    for graph_idx in range(num_graphs_to_plot):
        clean_graph = nx_graphs_to_plot[-1][graph_idx]
        pos = nx.spring_layout(clean_graph)
        for step_idx in range(num_timesteps):
            if graph_idx == 0:
                axes[graph_idx, step_idx].set_title(
                    f"t={np.round(timesteps[step_idx].item(), 2)}"
                )
            nx.draw(
                nx_graphs_to_plot[step_idx][graph_idx],
                ax=axes[graph_idx, step_idx],
                pos=pos,
            )


class QuantitativeResults:
    def __init__(self, dict_stat_fn, train_dataset, test_dataset):
        self.dict_stat_fn = dict_stat_fn
        self.stats = list(self.dict_stat_fn.keys())  # get names
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Get MMD(train, test)
        self.ref_metrics = {}
        for stat in self.stats:
            self.ref_metrics[stat] = self.compute_stat_mmd(
                train_dataset, test_dataset, dict_stat_fn[stat]
            )
        # print("Reference MMDs with the training dataset: ", self.ref_metrics)

        ref = {stat: 1 for stat in self.stats}
        self.results = {'ref': ref}

    def compute_normalized_stat_dict(self, dist):
        output_dict = {}
        for stat in self.stats:
            stat_fn = self.dict_stat_fn[stat]
            stat_mmd = self.compute_stat_mmd(dist, self.test_dataset, stat_fn)
            normalized_stat_mmd = stat_mmd / self.ref_metrics[stat]
            output_dict[stat] = normalized_stat_mmd

        return output_dict

    def compute_stat_mmd(self, dist1, dist2, stat_fn):
        deg_hist_1 = stat_fn(dist1)
        deg_hist_2 = stat_fn(dist2)
        return self.mmd(deg_hist_1, deg_hist_2)

    def mmd(self, X, Y, degree=2, gamma=1, coef0=0):
        XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def add_results(self, name, dist):
        self.results[name] = self.compute_normalized_stat_dict(dist)

    def show_table(self):
        return pd.DataFrame(self.results)



def nx_list_to_dataloader(nx_graph_list, bs):
    # Create empty lists to store node features, edge indices, and edge features
    data_list = []
    n_nodes_acc = 0

    # Iterate over each NetworkX graph
    for graph in nx_graph_list:
        # Extract node features from the graph
        cur_data = Data(x=torch.Tensor(list(graph.nodes())),
                    edge_index=torch.Tensor(list(graph.edges())).long().T,
                    edge_attr=torch.ones(len(list(graph.edges()))),
                    y=torch.tensor(0),
                    n_nodes=torch.Tensor([len(list(graph.nodes()))]))
        cur_data.x = torch.ones_like(cur_data.x)
        data_list.append(cur_data)
    
    return DataLoader(data_list, batch_size=bs, shuffle=True)

def symmetrize(E):
    # symmetrize the edge matrix - this function masks the diagonal of E at the same time
    upper_triangular_mask = torch.zeros_like(E).to(E.device)
    indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1).to(E.device)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1
    E = E * upper_triangular_mask
    E = E + torch.transpose(E, 1, 2)

    return E

def seed_everything(seed=42):
    # Seed python RNG
    random.seed(seed)
    
    # Seed numpy RNG
    np.random.seed(seed)
    
    # Seed torch RNG
    torch.manual_seed(seed)
    
    # If you are using CUDA:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure PyTorch to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False