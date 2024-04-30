import torch
import networkx as nx

from torch import Tensor


from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch_geometric as pyg


class PlaceHolder:
    def __init__(self, X: Tensor, E: Tensor, y: Tensor):
        self.X = X
        self.E = E
        self.y = y

    def to(self, device: str):
        X = self.X.to(device)
        E = self.E.to(device)
        y = self.y.to(device) if self.y is not None else None

        return PlaceHolder(X=X, E=E, y=y)

    def mask(self, node_mask):
        bs = node_mask.shape[0]
        n = node_mask.shape[1]
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1

        # Your solution here ###########################################################
        e_mask1 = ...  # bs, n, 1, 1
        e_mask2 = ...  # bs, 1, n, 1
        diag_mask = ...  # bs, n, n, 1
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.X = self.X * x_mask
        self.E = self.E * e_mask1 * e_mask2 * diag_mask
        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        return self

    def get_num_nodes(self):
        return torch.sum(
            self.X[:, :, 0] >= 1, dim=1
        )  # covers both collapsed and not collapsed

    def to_nx_graph_list(self):
        nx_graphs_list = []
        num_graphs = self.X.size(0)
        num_nodes = self.get_num_nodes()
        for graph_idx in range(num_graphs):
            num_nodes_graph = num_nodes[graph_idx].item()
            E_non_masked = self.E[graph_idx, :num_nodes_graph, :num_nodes_graph]
            E_collapsed = torch.argmax(E_non_masked, dim=-1)
            nx_graph = nx.from_numpy_array(E_collapsed.detach().cpu().numpy())
            nx_graphs_list.append(nx_graph)

        return nx_graphs_list

    def __repr__(self):
        return f"PlaceHolder(X={self.X.shape if self.X is not None else None}, E={self.E.shape if self.E is not None else None}, y={self.y.shape if self.y is not None else None})"


def er_validation_step(val_dataloader, val_metric, er_model):

    loss_list = []
    for batch in val_dataloader:
        # Access data
        unmasked_holder, node_mask = to_dense(batch)
        target_holder = unmasked_holder.mask(node_mask)

        # get prediction for er model
        sizes = target_holder.E.shape[:-1]
        er_E_pred = torch.tensor([1 - er_model.p, er_model.p]).repeat(*sizes, 1)
        pred = PlaceHolder(X=target_holder.X, E=er_E_pred, y=target_holder.y).mask(
            node_mask
        )
        E_true = target_holder.E.reshape(-1, target_holder.E.shape[-1])
        E_pred = pred.E.reshape(-1, pred.E.shape[-1])
        loss = val_metric(E_pred, E_true).detach().cpu().numpy()
        loss_list.append(loss)

    val_loss = sum(loss_list) / len(loss_list)

    return val_loss


def to_dense(data_batch, E_num_classes=2):
    x = data_batch.x.unsqueeze(-1)
    edge_index = data_batch.edge_index
    edge_attr = data_batch.edge_attr.unsqueeze(-1)
    batch = data_batch.batch
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=20)
    assert not pyg.utils.contains_self_loops(edge_index)  # in case, use line above
    E = to_dense_adj(
        edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=20
    )
    E = E + torch.transpose(E, 1, 2)

    E = torch.nn.functional.one_hot(E.long().squeeze(-1), E_num_classes).float()
    unmasked_holder = PlaceHolder(
        X=X, E=E, y=None
    )  # not masking because then we are sampling from multinomial

    return unmasked_holder, node_mask
