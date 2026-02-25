from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def get_models():

    return {
        "CatBoost": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            verbose=0
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=50, epsilon=0.1))
        ])
    }
