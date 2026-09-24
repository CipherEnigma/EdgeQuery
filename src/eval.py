import numpy as np


def auc(accuracy_curve):
    """Area under the accuracy-vs-budget-spent curve (trapezoidal rule)."""
    x = np.arange(len(accuracy_curve))
    return float(np.trapz(accuracy_curve, x))


def normalized_advantage(final_acc, observed_acc, full_acc, eps=1e-6):
    """Fraction of the achievable headroom (full-graph accuracy minus what the model
    already had before spending any budget) that a strategy's final accuracy captured.
    1.0 = reached full-graph accuracy using only the budget; 0.0 = no progress; can go
    negative if the strategy's picks actively hurt. NaN when there's ~no headroom to
    measure against (observed_acc already ~= full_acc)."""
    denom = full_acc - observed_acc
    if denom < eps:
        return float("nan")
    return (final_acc - observed_acc) / denom


def mean_and_se(curves):
    """curves: array of shape (n_seeds, n_steps). Returns (mean, standard_error) per step."""
    mean = curves.mean(axis=0)
    if curves.shape[0] > 1:
        se = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
    else:
        se = np.zeros_like(mean)
    return mean, se


def plot_curves(results, save_path, title="Test accuracy vs. budget spent"):
    """results: dict of {method_name: curves array of shape (n_seeds, n_steps)}."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for name, curves in results.items():
        mean, se = mean_and_se(curves)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=name, marker="o", markersize=3)
        plt.fill_between(x, mean - se, mean + se, alpha=0.2)

    plt.xlabel("Loop step (edges revealed)")
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
