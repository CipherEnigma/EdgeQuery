import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A):
    n = A.shape[0]
    A_hat = A + np.eye(n, dtype=A.dtype)
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm.astype(np.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A_norm):
        return A_norm @ self.linear(X)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, X, A_norm):
        h = self.layer1(X, A_norm)
        h = F.relu(h)
        out = self.layer2(h, A_norm)
        return out

def split_nodes(n, train_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)


def train_model(model, X, A_norm, y, train_idx, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X, A_norm)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
    return model


def accuracy(model, X, A_norm, y, idx):
    model.eval()
    with torch.no_grad():
        out = model(X, A_norm)
        preds = out[idx].argmax(dim=1)
        return (preds == y[idx]).float().mean().item()


def entropy_uncertainty(model, X, A_norm):
    model.eval()
    with torch.no_grad():
        out = model(X, A_norm)
        probs = F.softmax(out, dim=1)
        return -(probs * torch.log(probs + 1e-12)).sum(dim=1)


if __name__ == "__main__":
    from data import make_csbm

    y_np, A_np, X_np = make_csbm()
    n, feat_dim = X_np.shape
    num_classes = int(y_np.max()) + 1

    y = torch.tensor(y_np, dtype=torch.long)
    X = torch.tensor(X_np, dtype=torch.float32)
    train_idx, test_idx = split_nodes(n, train_frac=0.5, seed=0)

    A_norm_full = torch.tensor(normalize_adj(A_np), dtype=torch.float32)
    A_norm_none = torch.tensor(normalize_adj(np.zeros_like(A_np)), dtype=torch.float32)

    torch.manual_seed(0)
    model_full = GCN(feat_dim, 16, num_classes)
    train_model(model_full, X, A_norm_full, y, train_idx)
    acc_full = accuracy(model_full, X, A_norm_full, y, test_idx)

    torch.manual_seed(0)
    model_feat_only = GCN(feat_dim, 16, num_classes)
    train_model(model_feat_only, X, A_norm_none, y, train_idx)
    acc_feat_only = accuracy(model_feat_only, X, A_norm_none, y, test_idx)

    print(f"Full-graph test accuracy: {acc_full:.3f}")
    print(f"Features-only test accuracy: {acc_feat_only:.3f}")
    print(f"Headroom (gap): {acc_full - acc_feat_only:.3f}")



