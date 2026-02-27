import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="r2"
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, test_mean, label="Test Score")
    plt.title(f"Learning Curve for {name}")
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"outputs/figures/LearningCurve_{name}.jpg")
    plt.close()
