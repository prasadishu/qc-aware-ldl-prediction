import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
def evaluate(y_true, y_pred):
return {
r^2 = r^2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
pcc = np.corrcoef(y_true, y_pred)[0, 1]
}
   
