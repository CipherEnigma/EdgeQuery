import numpy as np


def auc(accuracy_curve):
    """Area under the accuracy-vs-budget-spent curve (trapezoidal rule)."""
    x = np.arange(len(accuracy_curve))
    return float(np.trapz(accuracy_curve, x))


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
