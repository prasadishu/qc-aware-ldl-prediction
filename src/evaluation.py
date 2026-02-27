from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate(y_test, predictions):
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(y_test, predictions)[0, 1]
    return r2, rmse, mse, pcc
   
