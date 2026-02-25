import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def generate_performance_summary(df, prediction_columns,
                                 target_column="LDL_direct"):

    """
    Generates overall performance summary table
    and saves as Performance_Summary.csv
    """

    os.makedirs("results", exist_ok=True)

    results = []

    for col in prediction_columns:

        y_true = df[target_column]
        y_pred = df[col]

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        pcc = np.corrcoef(y_true, y_pred)[0, 1]

        results.append({
            "Model/Formula": col,
            "N": len(df),
            "R_squared": round(r2, 6),
            "RMSE": round(rmse, 4),
            "MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "PCC": round(pcc, 6)
        })

    summary_df = pd.DataFrame(results)

    # Sort by best R2
    summary_df = summary_df.sort_values(
        by="R_squared",
        ascending=False
    )

    summary_df.to_csv(
        "results/Performance_Summary.csv",
        index=False
    )

    print("✅ Performance_Summary.csv generated successfully.")

    return summary_df
