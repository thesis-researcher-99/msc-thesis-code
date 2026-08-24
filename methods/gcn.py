"""
Graph Convolutional Network classifier two-sample test.
 
Classifier-based two-sample test (Lopez-Paz & Oquab, 2017 C2ST framework):
trains a 3-layer GCN (GCNConv + GraphNorm, weighted/z-scored-degree node
features, real edge weights threaded through message passing where
present) to discriminate between the two groups directly from the raw
graphs, then feeds its predictions to a permutation test (see
testing/permutation_test.py) to obtain a p-value from the held-out
misclassification error. Unlike knn.py/kernel_svm.py, this operates on
the raw graphs (via graphs_to_pyg_weighted_normalized), not a
precomputed distance/kernel matrix.
 
Entry points:
  get_predictions(G_all, y, idx_train, idx_test, ...) -> (y_pred, y_test)
      Fixed train/test split -- fit once (with early stopping on training-
      loss plateau), predict once. Feeds into the generic permutation test.
  build_base_data(G_all) + fit_predict_full(base_data_list, y, ...) -> y_pred
      Resubstitution variant: features/edges precomputed once via
      build_base_data (label-independent), then fit_predict_full re-fits
      per permutation with only the labels changing.
Neither entry point computes a p-value itself -- see testing/permutation_test.py.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.norm import GraphNorm


def graphs_to_pyg_weighted_normalized(G_all, y):
    """Weighted degree, z-scored per graph, as the node feature -- generalizes
    the validated degree_normalized fix (BA / gamma2 sweeps) to also use real
    edge weights where present. Falls back to plain z-scored degree when no
    'weight' attribute exists (e.g. unweighted BA graphs, where every edge
    defaults to weight 1.0, so weighted degree == plain degree).

    edge_weight is built here and actually threaded through GCNConv in
    forward() below -- both node features AND message passing are
    weight-aware."""
    data_list = []
    for G, label in zip(G_all, y):
        weighted_degrees = np.array(
            [sum(w for _, _, w in G.edges(n, data='weight', default=1.0)) for n in G.nodes()],
            dtype=np.float32
        )
        mean, std = weighted_degrees.mean(), weighted_degrees.std()
        wd_norm = (weighted_degrees - mean) / std if std > 1e-8 else weighted_degrees - mean
        x = torch.tensor(wd_norm.reshape(-1, 1), dtype=torch.float)

        edges = list(G.edges())
        weights = np.array([G[u][v].get('weight', 1.0) for u, v in edges], dtype=np.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight])

        data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                               y=torch.tensor([label], dtype=torch.long)))
    return data_list


class GCN(torch.nn.Module):
    """GraphNorm-equipped architecture, with edge_weight threaded through
    every GCNConv layer -- both node features (weighted, z-scored degree)
    and message passing (real edge weights) are weight-aware."""

    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.norm3 = GraphNorm(hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x, batch).relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x, batch).relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = self.norm3(x, batch)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


def get_predictions(G_all, y, idx_train, idx_test, hidden_channels=16,
                     epochs=150, patience=15, tol=1e-4, seed=12345):
    """Train a GCN ONCE on the shared train split, predict ONCE on the shared
    test split. Uses weighted, z-scored degree features + edge_weight message
    passing throughout (see graphs_to_pyg_weighted_normalized).
    """
    torch.manual_seed(seed)

    data_list = graphs_to_pyg_weighted_normalized(G_all, y)
    train_data = [data_list[i] for i in idx_train]
    test_data = [data_list[i] for i in idx_test]

    # full-batch: one batch containing the whole train split
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

    num_node_features = data_list[0].x.shape[1]
    num_classes = len(np.unique(y))

    model = GCN(num_node_features, hidden_channels, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        for data in train_loader:   # exactly one iteration, full batch
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_weight)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss - tol:
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.eval()
    test_loader_ordered = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    with torch.no_grad():
        data = next(iter(test_loader_ordered))
        out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_weight)
        y_pred = out.argmax(dim=-1).numpy()

    y_true_test = y[idx_test]
    return y_pred, y_true_test

def build_base_data(G_all):
    """Precompute node features + edge structure ONCE, label-independent.
    Reused across every permutation -- fit_predict_full only swaps in new
    labels, avoiding redundant feature/edge recomputation per call."""
    dummy_y = np.zeros(len(G_all), dtype=int)
    return graphs_to_pyg_weighted_normalized(G_all, dummy_y)


def fit_predict_full(base_data_list, y, hidden_channels=16, epochs=150,
                      patience=15, tol=1e-4, seed=12345):
    """Fit on ALL graphs with the given labels, predict on the SAME all
    graphs (resubstitution). base_data_list comes from build_base_data()."""
    torch.manual_seed(seed)
    y = np.asarray(y)

    data_list = []
    for i, data in enumerate(base_data_list):
        d = data.clone()
        d.y = torch.tensor([y[i]], dtype=torch.long)
        data_list.append(d)

    loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
    num_node_features = data_list[0].x.shape[1]
    num_classes = len(np.unique(y))

    model = GCN(num_node_features, hidden_channels, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    epochs_no_improve = 0
    model.train()
    for epoch in range(epochs):
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_weight)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        current_loss = loss.item()
        if current_loss < best_loss - tol:
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.eval()
    with torch.no_grad():
        data = next(iter(DataLoader(data_list, batch_size=len(data_list), shuffle=False)))
        out = model(data.x, data.edge_index, data.batch, edge_weight=data.edge_weight)
        y_pred = out.argmax(dim=-1).numpy()
    return y_pred
