import os
import pandas as pd

from data_loader import load_dataset
from formulas import friedewald, martin, sampson
from models import get_models
from qc_module import apply_qc
from evaluation import evaluate
from plots import scatter_plot
from performance_summary import generate_performance_summary
from tg_stratified import tg_stratified_analysis
from learning_curve_module import generate_learning_curves
from bland_altman import bland_altman_plot
from performance_summary import generate_performance_summary



# -------------------------------------------------
# Create result directories
# -------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# -------------------------------------------------
# Load datasets
# -------------------------------------------------
internal_df = load_dataset("data/LDL_Internal_Training_Test_Dataset1.xlsx")
secondary_df = load_dataset("data/LDL_Secondary_Internal_Validation_Dataset1.xlsx")


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

 prediction_cols = [
    "Friedewald",
    "Martin",
    "Sampson",
    "CatBoost",
    "XGBoost",
    "RandomForest",
    "SVR"
]


for name, model in models.items():
    generate_learning_curves(model, X, y, name)

    # Generate scatter plot
    preds_dict = {col: df[col] for col in ["Friedewald", "Martin", "Sampson"] + list(models.keys())}
    scatter_plot(y, preds_dict, f"Scatter_{dataset_name}.jpg")


# -------------------------------------------------
# Run processing
# -------------------------------------------------
process_dataset(LDL_Internal_Training_Test_Dataset1_df, "LDL_Internal_Training_Test_Dataset1")
process_dataset(LDL_Secondary_Internal_Validation_Dataset1_df, "LDL_Secondary_Internal_Validation_Dataset1")
generate_performance_summary(LDL_Internal_Training_Test_Dataset1_df, prediction_cols)
tg_stratified_analysis(LDL_Internal_Training_Test_Dataset1_df, prediction_cols)
bland_altman_plot(
    LDL_Internal_Training_Test_Dataset1_df["LDL_direct"],
    LDL_Internal_Training_Test_Dataset1_df["RandomForest"],
    "RandomForest"
)
models = get_models()
X = LDL_Internal_Training_Test_Dataset1_df[["TC", "TG", "HDL_C"]]
y = LDL_Internal_Training_Test_Dataset1_df["LDL_direct"]

print("\n🎯 All done successfully.")
