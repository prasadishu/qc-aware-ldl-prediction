import matplotlib.pyplot as plt
import os

def ensure_dir():
    os.makedirs("results/figures", exist_ok=True)

def scatter_plot(y_true, predictions, filename):

    ensure_dir()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (name, y_pred) in zip(axes, predictions.items()):

        ax.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Direct LDL-C")
        ax.set_ylabel("Predicted LDL-C")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}", dpi=300)
    plt.close()
