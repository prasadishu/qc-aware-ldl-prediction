import numpy as np
import matplotlib.pyplot as plt
import os


def bland_altman_plot(y_true, y_pred, model_name):

    os.makedirs("results/figures", exist_ok=True)

    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(6, 5))
    plt.scatter(mean_vals, diff_vals, alpha=0.6)
    plt.axhline(mean_diff, linestyle="--", label="Mean Bias")
    plt.axhline(upper, linestyle="--", label="+1.96 SD")
    plt.axhline(lower, linestyle="--", label="-1.96 SD")

    plt.xlabel("Mean of Direct & Predicted LDL-C")
    plt.ylabel("Difference (Predicted - Direct)")
    plt.title(f"Bland–Altman Plot - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/figures/BlandAltman_{model_name}.jpg", dpi=300)
    plt.close()

    print(f"✅ Bland–Altman plot saved for {model_name}")
