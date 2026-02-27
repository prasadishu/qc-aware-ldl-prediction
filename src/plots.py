import matplotlib.pyplot as plt

def actual_vs_predicted(direct, predicted, name):
    plt.figure()
    plt.scatter(direct, predicted, alpha=0.5)
    plt.plot([direct.min(), direct.max()],
             [direct.min(), direct.max()],
             linestyle="--")
    plt.xlabel("Direct LDL-C")
    plt.ylabel("Predicted LDL-C")
    plt.title(f"{name} Model")
    plt.savefig(f"outputs/figures/Actual_vs_Predicted_{name}.jpg")
    plt.close()

def residual_vs_predicted(predicted, residuals, name):
    plt.figure()
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted LDL-C")
    plt.ylabel("Residuals")
    plt.title(f"Residual vs Predicted - {name}")
    plt.savefig(f"outputs/figures/Residual_vs_Predicted_{name}.jpg")
    plt.close()
