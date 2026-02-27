import numpy as np

def apply_qc(predictions, direct_ldl):
    residuals = predictions - direct_ldl
    sd = np.std(residuals)
    qc_predictions = predictions.copy()

    for i in range(len(residuals)):
        if abs(residuals[i]) > 3 * sd:
            qc_predictions[i] = direct_ldl.iloc[i]

    return qc_predictions
