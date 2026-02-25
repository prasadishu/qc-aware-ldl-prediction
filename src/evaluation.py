def evaluate(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(y_true, y_pred)[0, 1]

    return r2, rmse, mse, pcc
  r2, rmse, mse, pcc = evaluate(y_test, predictions)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MSE: {mse:.3f}")
print(f"PCC: {pcc:.3f}"
