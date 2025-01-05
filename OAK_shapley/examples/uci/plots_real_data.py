import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_openml, fetch_california_housing
import sys
import os
# sys.path.append("/root/orthogonal-additive-gaussian-processes")

# Load datasets with parser set to 'auto' to avoid warnings
datasets = {
    "diabetes": load_diabetes(),
    "california_housing": fetch_california_housing(),
    "boston_housing": fetch_openml(data_id=531, as_frame=False, parser='auto'),
    "cpu_performance": fetch_openml(name="cpu", version=1, as_frame=False, parser='auto'),
    "wine_quality": fetch_openml(name="wine-quality-red", as_frame=False, parser='auto')
}
# Base directory containing the predictions
base_dir = "/root/OAK_shapley"

# Directory to save plots
plots_dir = os.path.join(base_dir, "real_data_plots")
os.makedirs(plots_dir, exist_ok=True)
# Iterate through datasets
for dataset_name in datasets.keys():
    try:
        # Load predictions from the CSV files
        predictions_oak_df = pd.read_csv(f"/root/orthogonal-additive-gaussian-processes/real_data_oak/predictions_oak_{dataset_name}.csv")
        predictions_kernel_df = pd.read_csv(f"/root/orthogonal-additive-gaussian-processes/real_data_shap/predictions_KernelExplainer_RandomForest_{dataset_name}.csv")
        predictions_mlp_df = pd.read_csv(f"/root/orthogonal-additive-gaussian-processes/real_data_shap/predictions_MLPRegressort_{dataset_name}.csv")
        predictions_additive_df = pd.read_csv(f"/root/orthogonal-additive-gaussian-processes/real_data_shap/predictions_AdditiveExplainer_{dataset_name}.csv")
        predictions_linearexpleiner_df = pd.read_csv(f"/root/orthogonal-additive-gaussian-processes/real_data_shap/predictions_LinearExplainer_{dataset_name}.csv")
        predictions_tree_df = pd.read_csv(f"/root/orthogonal-additive-gaussian-processes/real_data_shap/predictions_TreeExplainer_{dataset_name}.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure that the prediction CSV files for {dataset_name} are available.")
        continue  # Skip this iteration if any file is missing

    # Extract Iteration and Prediction columns
    iterations = predictions_oak_df["Iteration"]
    predictions_oak = predictions_oak_df["Prediction_OAK"]
    predictions_kernel = predictions_kernel_df["Predictions"]
    predictions_mlp = predictions_mlp_df["Predictions"]
    predictions_additive = predictions_additive_df["Predictions"]
    predictions_linear = predictions_linearexpleiner_df["Predictions"]
    predictions_tree = predictions_tree_df["Predictions"]

    # Plotting the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, predictions_oak, marker='o', linestyle='-', label='OAK iwth Shapley values Predictions')
    plt.plot(iterations, predictions_kernel[:len(iterations)], marker='x', linestyle='--', label='Random Forest withKernel ex. Predictions')
    plt.plot(iterations, predictions_mlp[:len(iterations)], marker='s', linestyle='-.', label='MLPRegressor with Kernel ex.Predictions')
    plt.plot(iterations, predictions_additive[:len(iterations)], marker='^', linestyle='-', label='Linear model Additive ex. Predictions')
    plt.plot(iterations, predictions_linear[:len(iterations)], marker='^', linestyle='-', label='Linear model Linear ex. Predictions')
    plt.plot(iterations, predictions_tree[:len(iterations)], marker='^', linestyle='-', label='Tree model with Tree ex. Predictions')

    plt.xlabel('Iteration')
    plt.ylabel('Predictions')
    plt.title(f'Model Predictions over Iterations on {dataset_name} dataset')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    file_path = os.path.join(plots_dir, f"{dataset_name}_predictions.png")
    plt.savefig(file_path)
    plt.close()

