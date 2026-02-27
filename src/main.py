import os
import pandas as pd
from data_loader import load_dataset
from formulas import friedewald, martin, sampson
from models import get_models
from evaluation import evaluate
from qc_module import apply_qc
from learning_curve_module import plot_learning_curve
from plots import actual_vs_predicted, residual_vs_predicted
from bland_altman import bland_altman_plot
from performance_summary import save_performance

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# Load Data
df = load_dataset("data/LDL_Internal_Training_Test_Dataset1.xlsx")

X = df[["TC", "HDL", "TG"]]
y = df["Direct_LDL"]

models = get_models()
results = []

for name, model in models.items():
    model.fit(X, y)
    predictions = model.predict(X)

    r2, rmse, mse, pcc = evaluate(y_test, predictions)
    results.append({
        "Model": name,
        "R2": r2,
        "RMSE": rmse,
        "MSE": mse,
        "PCC": pcc
    })

    qc_predictions = apply_qc(predictions, y)

    actual_vs_predicted(y, predictions, name)
    residual_vs_predicted(predictions, predictions - y, name)
    plot_learning_curve(model, X, y, name)
    bland_altman_plot(y, qc_predictions)

# Formula Predictions
df["Friedewald"] = friedewald(df["TC"], df["HDL"], df["TG"])
df["Martin"] = martin(df["TC"], df["HDL"], df["TG"])
df["Sampson"] = sampson(df["TC"], df["HDL"], df["TG"])

save_performance(results)

print("Pipeline executed successfully.")
