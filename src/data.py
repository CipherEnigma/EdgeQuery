import numpy as np
# building our own gcn model to prove its learning from graph connectivity and not just cheating off easy features

# params:
# n: total number of nodes in the graph
# k: number of classes/communities
# p_in: probab of two nodes getting an edge
# p_out: probab of an edge if two nodel are of different classes, much lower, pin>>>pout
# feat_dim: length of each node's feat vec
# feat_signal: this is basically used to determine that there are faint signals in the features and can carry some weak correleation, not enough by itself.
# if we use zero signal--> then the edge would be unrealisitic,

def make_csbm(n=90, k=3, p_in=0.11, p_out=0.005, feat_dim=12, feat_signal=0.10, seed=0):
    rng = np.random.default_rng(seed)

    base = n // k
    remainder = n - base * k
    y = np.repeat(np.arange(k), base)
    if remainder:
        y = np.concatenate([y, np.arange(remainder)])
    y = y.astype(np.int64)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if rng.random() < p:
                A[i, j] = 1.0
                A[j, i] = 1.0
    
    centers = rng.normal(size=(k, feat_dim))
    X = rng.normal(size=(n, feat_dim)) + feat_signal * centers[y]
    X = X.astype(np.float32)

    return y, A, X

if __name__ == "__main__":
    y, A, X = make_csbm()

    counts = np.bincount(y)
    print("Node count per class:", counts.tolist())

    iu = np.triu_indices(len(y), k=1)
    edge_mask = A[iu] == 1.0
    same_class = y[iu[0]] == y[iu[1]]
    within = int(np.sum(edge_mask & same_class))
    between = int(np.sum(edge_mask & ~same_class))

    print("Within-class edges:", within)
    print("Between-class edges:", between)
    print("Total edges:", within + between)







