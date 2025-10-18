# Lyft_dataset_new_pipeline_4_Grid_Search_.py
import os
# Disable MLflow auto emoji logging before importing mlflow
os.environ["MLFLOW_DISABLE_AUTO_LOG_EMOJIS"] = "1"

import sys
import mlflow
from mlflow.tracking import MlflowClient

# Override MlflowClient._log_url to avoid UnicodeEncodeError due to emojis
def safe_log_url(self, run_id):
    try:
        run_info = self.get_run(run_id).info
        run_url = f"{mlflow.get_tracking_uri()}/#/experiments/{run_info.experiment_id}/runs/{run_id}"
        sys.stdout.write(f"View run at: {run_url}\n")
    except Exception:
        pass

experiment_name = "Lyft_Model_Experiment_Grid_Bayesian_Search_1"
experiment = mlflow.get_experiment_by_name(experiment_name)

with mlflow.start_run(run_name="Initial_Run_Info") as run:
    run_id = run.info.run_id
    experiment_id = experiment.experiment_id
    print(f"View run at: http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}")



# Now safe to import mlflow submodules
import mlflow.sklearn
from mlflow import log_params

# Standard imports
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Optional: Bayesian optimization imports
from skopt import BayesSearchCV
from skopt.space import Integer

# --- MLflow setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lyft_Model_Experiment_Grid_Bayesian_Search_2")

# --- Load dataset ---
df = pd.read_csv("Lyftdataset.csv")

# Load and apply metadata types
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
print("Dataset preview:")
print(df.head())

# --- Prepare features and target ---
y = df[["cnt"]]
X = df.drop(["instant", "dteday", "casual", "registered", "cnt"], axis=1)

# --- Split train-test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features preview:")
print(X.head())
print("Target preview:")
print(y.head())

# --- Define feature categories ---
ordinal_cols = ["weathersit"]
nominal_cols = [col for col in X.select_dtypes(include="category").columns if col not in ordinal_cols]
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

# --- Build preprocessing pipelines ---
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

# Fit preprocessor to training data
processed_train = preprocessor.fit_transform(X_train)
print(processed_train.head())

joblib.dump(preprocessor, "preprocessor_pipeline.pkl")
print("Saved preprocessing pipeline to preprocessor_pipeline.pkl")




# --- Create model pipeline ---
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

# Log Grid Search CV results to MLflow with nested runs
for i, candidate in enumerate(grid_search_cv.cv_results_["params"]):
    with mlflow.start_run(nested=True, run_name=f"GridSearch_CV_Set_{i}"):
        log_params(candidate)
        mean_score = grid_search_cv.cv_results_["mean_test_score"][i]
        std_score = grid_search_cv.cv_results_["std_test_score"][i]
        mlflow.log_metric("cv_mean_score", mean_score)
        mlflow.log_metric("cv_std_score", std_score)

print("Best Parameters (Grid Search):", grid_search_cv.best_params_)
print("Best CV Score (MSE):", grid_search_cv.best_score_)

# --- Evaluate best model ---
best_model = grid_search_cv.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(best_model, "grid_search_best_model.pkl")
print("Saved best Grid Search model to grid_search_best_model.pkl")


# --- Log Grid Search best model to MLflow with signature ---
import mlflow.models

rmse = np.sqrt(mse)  # Calculate RMSE

with mlflow.start_run(run_name="GridSearch_DT"):
    mlflow.log_params(grid_search_cv.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("best_cv_score", grid_search_cv.best_score_)

    # Log model with input/output signature
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(best_model, "model", signature=signature)

    print("Logged Grid Search Run to MLflow")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
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

# Log Bayesian CV results to MLflow with nested runs
for i, candidate in enumerate(bayes_search_cv.cv_results_["params"]):
    with mlflow.start_run(nested=True, run_name=f"BayesSearch_CV_Set_{i}"):
        log_params(candidate)
        mean_score = bayes_search_cv.cv_results_["mean_test_score"][i]
        std_score = bayes_search_cv.cv_results_["std_test_score"][i]
        mlflow.log_metric("cv_mean_score", mean_score)
        mlflow.log_metric("cv_std_score", std_score)

print("Best Parameters (Bayesian Search):", bayes_search_cv.best_params_)
print("Best Score (Bayesian Search MSE):", bayes_search_cv.best_score_)

# --- Evaluate Bayesian best model ---
y_pred_bayes = bayes_search_cv.best_estimator_.predict(X_test)
mse_bayes = mean_squared_error(y_test, y_pred_bayes)
r2_bayes = r2_score(y_test, y_pred_bayes)

joblib.dump(bayes_search_cv.best_estimator_, "bayes_search_best_model.pkl")
print("Saved best Bayesian Search model to bayes_search_best_model.pkl")


print(f"Bayesian Search Test MSE: {mse_bayes:.4f}")
print(f"Bayesian Search Test R2 Score: {r2_bayes:.4f}")

# --- Log Bayesian Search best model to MLflow with signature ---
import mlflow.models

rmse_bayes = np.sqrt(mse_bayes)  # Calculate RMSE

with mlflow.start_run(run_name="BayesSearch_DT"):
    mlflow.log_params(bayes_search_cv.best_params_)
    mlflow.log_metric("mse", mse_bayes)
    mlflow.log_metric("rmse", rmse_bayes)
    mlflow.log_metric("r2", r2_bayes)
    mlflow.log_metric("best_cv_score", bayes_search_cv.best_score_)

    # Log model with input/output signature
    signature = mlflow.models.infer_signature(X_test, y_pred_bayes)
    mlflow.sklearn.log_model(bayes_search_cv.best_estimator_, "model", signature=signature)

    print("Logged Bayesian Search Run to MLflow")
    print(f"Test MSE: {mse_bayes:.4f}")
    print(f"Test RMSE: {rmse_bayes:.4f}")
    print(f"Test R2 Score: {r2_bayes:.4f}")

# --- Function to plot metric boxplot for an MLflow experiment ---
def plot_mlflow_metric_boxplot(experiment_name, metric_name="mse"):
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )

    metric_col = f"metrics.{metric_name}"
    if metric_col not in runs.columns:
        print(f"Metric '{metric_name}' not found in runs.")
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(runs[metric_col].dropna(), vert=False)
    plt.title(f"Distribution of '{metric_name}' Across MLflow Runs")
    plt.xlabel(metric_name)
    plt.grid(True)
    plt.show()


# Plot MSE box plot
plot_mlflow_metric_boxplot("Lyft_Model_Experiment_2", metric_name="mse")


# Plot RMSE box plot
plot_mlflow_metric_boxplot("Lyft_Model_Experiment_2", metric_name="rmse")

# --- Final Summary ---
print("\n--- Final Model Performance Summary ---")
print(f"Grid Search -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
print(f"Bayesian Search -> MSE: {mse_bayes:.4f}, RMSE: {rmse_bayes:.4f}, R2: {r2_bayes:.4f}")
