# Lyft_dataset_XGBoost_predicting_Final_model.py

import joblib
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Load the trained model pipeline ===
loaded_model = joblib.load("xgboost_best_model.pkl")

# === Re-load dataset and metadata ===
df = pd.read_csv("lyftdataset.csv")

# Load and apply data types from metadata
with open("lyftdataset_metadata.json", "r") as f:
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

# === Prepare features and target ===
X = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
y = df["cnt"]

# === Recreate the same train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Predict using the loaded model ===
y_pred = loaded_model.predict(X_test)

# === Evaluate model performance ===
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === Save predictions to CSV ===
results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = y_pred
results_df.to_csv("xgboost_predictions.csv", index=False)
print("📁 Predictions saved to 'xgboost_predictions.csv'")



import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("file:///C:/Users/charles/Desktop/Data Scientist/SDSU/BDA_602/Lyft Inc Dataset/Lyft_GitHub/BDA602/mlruns")# Optional, if not already set
mlflow.set_experiment("XGBoost_Regression_Final_Eval")

with mlflow.start_run(run_name="XGBoost Predictions Evaluation"):
    # --- Plot 1: Scatter Plot (Predicted vs Actual) ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("scatter_actual_vs_predicted.png")
    plt.close()

    # --- Plot 2: Line Plot (Compare Actual and Predicted) ---
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:100], label="Actual")
    plt.plot(y_pred[:100], label="Predicted")
    plt.legend()
    plt.title("Line Plot of Actual vs Predicted (First 100)")
    plt.tight_layout()
    plt.savefig("lineplot_actual_vs_predicted.png")
    plt.close()

    # --- Plot 3: Residuals Distribution ---
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.savefig("residual_distribution.png")
    plt.close()

    # ✅ Log Artifacts to MLflow
    mlflow.log_artifact("scatter_actual_vs_predicted.png")
    mlflow.log_artifact("lineplot_actual_vs_predicted.png")
    mlflow.log_artifact("residual_distribution.png")

    # ✅ log final metrics too
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("final_rmse", rmse)
    mlflow.log_metric("final_r2", r2)



"""import matplotlib.pyplot as plt
import seaborn as sns
"""
# 1. Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Rental Count")
plt.ylabel("Predicted Rental Count")
plt.title("Actual vs Predicted Bike Rental Counts")
plt.tight_layout()
plt.savefig("scatter_actual_vs_predicted.png")
plt.show()

# 2. Line Plot: Side-by-side comparison (for first 100 samples)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label="Actual", marker='o')
plt.plot(y_pred[:100], label="Predicted", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Bike Rental Count")
plt.title("Actual vs Predicted (First 100 Samples)")
plt.legend()
plt.tight_layout()
plt.savefig("lineplot_actual_vs_predicted.png")
plt.show()

# 3. Residual Plot (errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig("residual_distribution.png")
plt.show()
