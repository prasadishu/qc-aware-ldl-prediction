from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(y_true, y_pred)[0, 1]

    return r2, rmse, mse, pcc
