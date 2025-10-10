# Lyft_dataset_pipeline module 4 Corrected for Regression
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# Sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# MLflow for tracking
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lyft_Model_Experiment")


# Optional: Bayesian optimization imports
from skopt import BayesSearchCV
from skopt.space import Integer

# --- Load dataset ---
df = pd.read_csv("Lyftdataset.csv")

# Load and apply metadata
with open("Lyftdataset_metadata.json", "r") as f:
    metadata = json.load(f)

for col, dtype in metadata["dtypes"].items():
    if dtype == 'int64':
        df[col] = df[col].astype('Int64')
    elif dtype == 'category':
        df[col] = df[col].astype('category')
    elif dtype == 'float64':
        df[col] = df[col].astype('float')
    elif dtype == 'bool':
        df[col] = df[col].astype('bool')
    elif dtype == 'object':
        df[col] = df[col].astype('object')

print(df.info())
print("This is the full head!")
print(df.head())

# --- Feature setup ---
y = df[["cnt"]]
X = df.drop(["instant", "dteday", "casual", "registered", "cnt"], axis=1)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("This is the cleaned head!")
print(X.head())
print(y.head())

# --- Feature type categorization ---
ordinal_cols = ["weathersit"]
nominal_cols = [col for col in X.select_dtypes(include="category").columns if col not in ordinal_cols]
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

# --- Preprocessing pipelines ---
ordinal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

nominal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])

preprocessor = ColumnTransformer([
    ("ordinal", ordinal_pipe, ordinal_cols),
    ("nominal", nominal_pipe, nominal_cols),
    ("numeric", numeric_pipe, numeric_cols)
])

preprocessor.set_output(transform="pandas")

processed_train = preprocessor.fit_transform(X_train)
print(processed_train.head())

# --- Model pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# --- Grid Search ---
hyperparam_grid = {
    "regressor__max_depth": [3, 5, 10],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4]
}

grid_search_cv = GridSearchCV(
    estimator=model,
    param_grid=hyperparam_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search_cv.fit(X_train, y_train.values.ravel())

print("Best Parameters (Grid Search):", grid_search_cv.best_params_)
print("Best CV Score (MSE):", grid_search_cv.best_score_)

# --- Evaluation (Grid Search) ---
best_model = grid_search_cv.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Log Grid Search to MLflow ---
with mlflow.start_run(run_name="GridSearch_DT"):
    mlflow.log_params(grid_search_cv.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(best_model, "model")
    print("Logged Grid Search Run to MLflow")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")

# --- Bayesian Optimization ---
search_space = {
    "regressor__max_depth": Integer(3, 20),
    "regressor__min_samples_split": Integer(2, 10),
    "regressor__min_samples_leaf": Integer(1, 5)
}

bayes_search_cv = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=30,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

bayes_search_cv.fit(X_train, y_train.values.ravel())

print("Best Parameters (Bayesian Search):", bayes_search_cv.best_params_)
print("Best Score (Bayesian Search MSE):", bayes_search_cv.best_score_)

# --- Evaluation (Bayesian Search) ---
y_pred_bayes = bayes_search_cv.best_estimator_.predict(X_test)
mse_bayes = mean_squared_error(y_test, y_pred_bayes)
r2_bayes = r2_score(y_test, y_pred_bayes)

print(f"Bayesian Search Test MSE: {mse_bayes:.4f}")
print(f"Bayesian Search Test R2 Score: {r2_bayes:.4f}")

# --- Log Bayesian Search to MLflow ---
with mlflow.start_run(run_name="BayesSearch_DT"):
    mlflow.log_params(bayes_search_cv.best_params_)
    mlflow.log_metric("mse", mse_bayes)
    mlflow.log_metric("r2", r2_bayes)
    mlflow.sklearn.log_model(bayes_search_cv.best_estimator_, "model")
    print("Logged Bayesian Search Run to MLflow")
    print(f"Test MSE: {mse_bayes:.4f}")
    print(f"Test R2 Score: {r2_bayes:.4f}")
