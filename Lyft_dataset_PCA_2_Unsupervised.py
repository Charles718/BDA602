# Lyft_dataset_PCA_2_Unsupervised.py
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow import log_params

# Disable MLflow emoji logging and patch unicode URL logging
os.environ["MLFLOW_DISABLE_AUTO_LOG_EMOJIS"] = "1"

def safe_log_url(self, run_id):
    try:
        run_info = self.get_run(run_id).info
        run_url = f"{mlflow.get_tracking_uri()}/#/experiments/{run_info.experiment_id}/runs/{run_id}"
        sys.stdout.write(f"View run at: {run_url}\n")
    except Exception:
        pass

MlflowClient._log_url = safe_log_url

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lyft_Unsupervised_PCA_KMeans_2")

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

preprocessor = joblib.load("preprocessor_pipeline.pkl")
df = pd.read_csv("lyftdataset.csv")
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

X = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
numeric_pipeline = dict(preprocessor.named_transformers_)["numeric"]
X_num = X.select_dtypes(include="number")

pca_model = Pipeline(steps=[("preprocessor", numeric_pipeline), ("pca", PCA())])
pca_model.fit(X_num)
pca_step = pca_model.named_steps["pca"]

# Directory to save plots temporarily
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

with mlflow.start_run(run_name="PCA_KMeans_Clustering"):

    mlflow.log_param("pca_n_components", pca_step.n_components_)
    total_explained_variance = np.sum(pca_step.explained_variance_ratio_)
    mlflow.log_metric("explained_variance_ratio_sum", total_explained_variance)
    for i, evr in enumerate(pca_step.explained_variance_ratio_):
        mlflow.log_metric(f"explained_variance_ratio_pc_{i+1}", evr)

    loadings_df = pd.DataFrame(
        pca_step.components_.T,
        index=X_num.columns.to_list(),
        columns=[f"PC{i+1}" for i in range(pca_step.n_components_)],
    )

    # Scree plot
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, pca_step.n_components_ + 1), pca_step.explained_variance_ratio_, marker='o', linestyle='-')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot: Variance Explained by Each Principal Component")
    plt.xticks(np.arange(1, pca_step.n_components_ + 1))
    plt.grid(True)
    plt.tight_layout()
    scree_path = os.path.join(plot_dir, "scree_plot.png")
    plt.savefig(scree_path)
    mlflow.log_artifact(scree_path)
    plt.close()

    # Cumulative variance plot
    cumulative_variance = np.cumsum(pca_step.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Variance Explained by PCA")
    plt.grid(True)
    plt.tight_layout()
    cumvar_path = os.path.join(plot_dir, "cumulative_variance_plot.png")
    plt.savefig(cumvar_path)
    mlflow.log_artifact(cumvar_path)
    plt.close()

    # Heatmap of top_k PCA loadings
    top_k = 5
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        loadings_df.iloc[:, :top_k], cmap="coolwarm", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7}
    )
    plt.title("PCA Loadings: Feature Contribution to Principal Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Original Features")
    plt.tight_layout()
    heatmap_path = os.path.join(plot_dir, "pca_loadings_heatmap.png")
    plt.savefig(heatmap_path)
    mlflow.log_artifact(heatmap_path)
    plt.close()

    # Prepare for clustering
    X_transformed = numeric_pipeline.fit_transform(X_num)
    k_range = range(2, 7)
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_transformed)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_transformed, kmeans.labels_)
        silhouette_scores.append(score)
        mlflow.log_metric(f"inertia_k_{k}", kmeans.inertia_)
        mlflow.log_metric(f"silhouette_score_k_{k}", score)

    print("Inertias:", inertias)
    print("Silhouette Scores:", silhouette_scores)

    best_k = k_range[np.argmax(silhouette_scores)]
    mlflow.log_param("best_k", best_k)

    # Plot inertia and silhouette score together
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia', color=color1)
    ax1.plot(list(k_range), inertias, marker='o', linestyle='-', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Silhouette Score', color=color2)
    ax2.plot(list(k_range), silhouette_scores, marker='s', linestyle='--', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Inertia and Silhouette Score vs. Number of Clusters (k)")
    fig.tight_layout()
    plt.grid(True)
    inertia_silhouette_path = os.path.join(plot_dir, "inertia_silhouette_plot.png")
    plt.savefig(inertia_silhouette_path)
    mlflow.log_artifact(inertia_silhouette_path)
    plt.close()

    # Final KMeans clustering & PCA 2D visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_transformed)
    best_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    best_kmeans.fit(X_transformed)
    labels = best_kmeans.labels_

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        c=labels, cmap='viridis', s=50, alpha=0.7
    )
    centers_2d = pca_2d.transform(best_kmeans.cluster_centers_)
    plt.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        c='red', marker='X', s=200, label='Centroids'
    )
    plt.title(f"KMeans Clustering (k={best_k}) in PCA-Reduced 2D Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cluster_plot_path = os.path.join(plot_dir, "kmeans_pca_2d_plot.png")
    plt.savefig(cluster_plot_path)
    mlflow.log_artifact(cluster_plot_path)
    plt.close()

    # Log models
    mlflow.sklearn.log_model(best_kmeans, artifact_path="kmeans_model")
    mlflow.sklearn.log_model(pca_model, artifact_path="pca_pipeline")
