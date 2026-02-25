import os
import pandas as pd

from data_loader import load_dataset
from formulas import friedewald, martin, sampson
from models import get_models
from qc_module import apply_qc
from evaluation import evaluate
from plots import scatter_plot


# -------------------------------------------------
# Create result directories
# -------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# -------------------------------------------------
# Load datasets
# -------------------------------------------------
internal_df = load_dataset("data/internal_dataset.xlsx")
secondary_df = load_dataset("data/secondary_dataset.xlsx")


def process_dataset(df, dataset_name):

    X = df[["TC", "TG", "HDL_C"]]
    y = df["LDL_direct"]

    # Formula predictions
    df["Friedewald"] = friedewald(df["TC"], df["TG"], df["HDL_C"])
    df["Martin"] = martin(df["TC"], df["TG"], df["HDL_C"])
    df["Sampson"] = sampson(df["TC"], df["TG"], df["HDL_C"])

    models = get_models()

    for name, model in models.items():

        model.fit(X, y)
        preds = model.predict(X)
        preds_qc = apply_qc(preds, y)

        df[name] = preds_qc

    # Save CSV
    output_path = f"results/predictions_{dataset_name}.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved: {output_path}")

    # Print metrics
    print(f"\n📊 Performance on {dataset_name} dataset:")

    for col in ["Friedewald", "Martin", "Sampson"] + list(models.keys()):
        r2, rmse, pcc = evaluate(y, df[col])
        print(f"{col}: R²={r2:.3f} | RMSE={rmse:.3f} | PCC={pcc:.3f}")

    # Generate scatter plot
    preds_dict = {col: df[col] for col in ["Friedewald", "Martin", "Sampson"] + list(models.keys())}
    scatter_plot(y, preds_dict, f"Scatter_{dataset_name}.jpg")


# -------------------------------------------------
# Run processing
# -------------------------------------------------
process_dataset(internal_df, "internal")
process_dataset(secondary_df, "secondary")

print("\n🎯 All done successfully.")
