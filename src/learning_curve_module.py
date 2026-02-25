import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import learning_curve


def generate_learning_curves(model, X, y, model_name):

    os.makedirs("results/figures", exist_ok=True)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, train_mean, label="Training R²")
    plt.plot(train_sizes, test_mean, label="Validation R²")

    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/figures/LearningCurve_{model_name}.jpg", dpi=300)
    plt.close()

    print(f"✅ Learning curve saved for {model_name}")
