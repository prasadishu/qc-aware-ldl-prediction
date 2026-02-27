from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def get_models():
    models = {
        "CatBoost": CatBoostRegressor(verbose=0),
        "XGBoost": XGBRegressor(),
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR(kernel="rbf")
    }
    return models
