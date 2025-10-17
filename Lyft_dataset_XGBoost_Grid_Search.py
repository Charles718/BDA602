#Lyft_dataset_XGBoost_Grid_Search.py
# Data processing libraries
import pandas as pd
import numpy as np
import scipy as sp
import json

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline serialization library (used for exports & imports)
import joblib


# Sklearn libraries for preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Manual Grid Search
from sklearn.model_selection import (
    train_test_split,
    ParameterGrid
)


# XGBoost Classifier Algorithm
from xgboost import XGBRegressor

# MLflow Library Imports and OS Directory Handling
import os
from mlflow import (
    set_tracking_uri, get_tracking_uri, set_experiment, start_run, 
    log_params, log_metric
)
from mlflow.models.signature import infer_signature
import mlflow.sklearn

# Metric to track and log for MLflow
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score


# Import entire serialized preprocessing pipeline
preprocessor = joblib.load("preprocessor_pipeline.pkl")

# Configure this week into central ML repository for course
#####set_tracking_uri("file:///mnt/c/Users/rpala/Desktop/BDA602/BDA602/mlruns")

# Set experiment name for this module
set_experiment("XGBoost_1")


df = pd.read_csv("lyftdataset.csv")
with open("lyftdataset_metadata.json", "r") as f:
    metadata = json.load(f)

# Apply data types from metadata
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


# Define predictors and target
X = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])  # predictors
y = df["cnt"]  # target

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Specify hyperparameters
hyperparam_grid = {
    "n_estimators": [50, 200],  # Total number of trees in ensemble
    "max_depth": [3, 10],     # Max tree depth
    "learning_rate": [0.05, 0.1, 0.5]   # Different learning rates
}


# Track best model
best_rmse = float("inf")
best_model = None
best_params = None



# Iterate through all hyperparameter configurations
for hyperparams in ParameterGrid(hyperparam_grid):
    with start_run(run_name="XGBoost Grid Search", nested=True):
        # Define XGBoost model in pipeline
        xgboost_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                eval_metric="rmse",  # appropriate for regression
                **hyperparams
            ))
        ])

        # Fit model and predict
        xgboost_model.fit(X_train, y_train)
        y_pred = xgboost_model.predict(X_test)

        # Evaluate regression metrics
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        # Track best model
        if rmse < best_rmse:
           best_rmse = rmse
           best_model = xgboost_model
           best_params = hyperparams


        # Log parameters and metrics
        log_params(hyperparams)
        log_metric("rmse", rmse)
        log_metric("r2", r2)

        # Log model
        input_example = pd.DataFrame(X_test[:1], columns=X_test.columns)
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            xgboost_model, artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.tight_layout()

        # Save and log the plot
        plot_filename = f"residuals_n{hyperparams['n_estimators']}_d{hyperparams['max_depth']}_lr{hyperparams['learning_rate']}.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        plt.close()

        # Get feature importances
        model_only = xgboost_model.named_steps["regressor"]
        importances = model_only.feature_importances_

        # Get preprocessed feature names
        feature_names = xgboost_model.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Save and log as artifact
        imp_file = f"feature_importance_n{hyperparams['n_estimators']}_d{hyperparams['max_depth']}_lr{hyperparams['learning_rate']}.csv"
        importance_df.to_csv(imp_file, index=False)
        mlflow.log_artifact(imp_file)


print("Best Model Parameters:")
print(best_params)
print(f"Best RMSE: {best_rmse:.4f}")

# Optional: evaluate again
y_pred_best = best_model.predict(X_test)
print(f"R² of Best Model: {r2_score(y_test, y_pred_best):.4f}")

with start_run(run_name="Best XGBoost Model"):
    mlflow.sklearn.log_model(best_model, "best_model")
    log_params(best_params)
    log_metric("best_rmse", best_rmse)
    log_metric("best_r2", r2_score(y_test, y_pred_best))
    print("Logged Best Model to MLflow")