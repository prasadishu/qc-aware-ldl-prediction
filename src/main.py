import os
import pandas as pd

from data_loader import load_dataset
from formulas import friedewald, martin, sampson
from models import get_models
from qc_module import apply_qc
from evaluation import evaluate

from performance_summary import generate_performance_summary
from tg_stratified import tg_stratified_analysis
from learning_curve_module import generate_learning_curves
from bland_altman import bland_altman_plot


# -------------------------------------------------------
# Create results directories
# -------------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------
internal_df = load_dataset("data/LDL_Internal_Training_Test_Dataset1.xlsx")
secondary_df = load_dataset("data/LDL_Secondary_Internal_Validation_Dataset1.xlsx")

r2, rmse, mse, pcc = evaluate(y_test, predictions)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MSE: {mse:.3f}")
print(f"PCC: {pcc:.3f}")
# -------------------------------------------------------
# FUNCTION TO PROCESS DATASET
# -------------------------------------------------------
def process_dataset(df, dataset_name):

    print(f"\n🔹 Processing {dataset_name} dataset")

    X = df[["TC", "TG", "HDL_C"]]
    y = df["LDL_direct"]

    # ---------------------------------------------------
    # 1️⃣ Calculate Formula-Based LDL
    # ---------------------------------------------------
    df["LDL_Friedewald"] = friedewald(df["TC"], df["TG"], df["HDL_C"])
    df["LDL_Martin"] = martin(df["TC"], df["TG"], df["HDL_C"])
    df["LDL_Sampson"] = sampson(df["TC"], df["TG"], df["HDL_C"])

    # ---------------------------------------------------
    # 2️⃣ Train ML Models + Apply QC
    # ---------------------------------------------------
    models = get_models()

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X, y)
        preds = model.predict(X)

        preds_qc = apply_qc(preds, y)

        df[f"LDL_{name}_QC"] = preds_qc

    # ---------------------------------------------------
    # 3️⃣ Save Prediction CSV
    # ---------------------------------------------------
    prediction_path = f"results/predictions_{dataset_name}.csv"
    df.to_csv(prediction_path, index=False)
    print(f"✅ Predictions saved → {prediction_path}")

    # ---------------------------------------------------
    # 4️⃣ Generate Performance Summary CSV
    # ---------------------------------------------------
    prediction_cols = [
        "LDL_RandomForest_QC",
        "LDL_XGBoost_QC",
        "LDL_CatBoost_QC",
        "LDL_SVR_QC",
        "LDL_Martin",
        "LDL_Friedewald",
        "LDL_Sampson"
    ]

    summary_df = generate_performance_summary(df, prediction_cols)

    print("\n📊 Performance Summary:")
    print(summary_df)

    # ---------------------------------------------------
    # 5️⃣ TG-Stratified Analysis
    # ---------------------------------------------------
    tg_stratified_analysis(df, prediction_cols)

    # ---------------------------------------------------
    # 6️⃣ Bland–Altman Plots (ML models only)
    # ---------------------------------------------------
    for col in prediction_cols:
        bland_altman_plot(
            df["LDL_direct"],
            df[col],
            f"{col}_{dataset_name}"
        )

    # ---------------------------------------------------
    # 7️⃣ Learning Curves (ML models only)
    # ---------------------------------------------------
    for name, model in models.items():
        generate_learning_curves(model, X, y, f"{name}_{dataset_name}")

    print(f"\n🎯 Completed {dataset_name} dataset.\n")


# -------------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------------
process_dataset(LDL_Internal_Training_Test_Dataset1_df, "LDL_Internal_Training_Test_Dataset1")
process_dataset(LDL_Secondary_Internal_Validation_Dataset1_df, "LDL_Secondary_Internal_Validation_Dataset1")

print("\n🚀 All modules executed successfully.")

