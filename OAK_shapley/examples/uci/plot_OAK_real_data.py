import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_openml, fetch_california_housing
import sys
import os
sys.path.append("/root/orthogonal-additive-gaussian-processes")

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
        predictions_oak_df = pd.read_csv(f"/root/OAK_shapley/real_data_oak/predictions_oak_{dataset_name}.csv")
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure that the prediction CSV files for {dataset_name} are available.")
        continue  # Skip this iteration if any file is missing

    # Extract Iteration and Prediction columns
    iterations = predictions_oak_df["Iteration"]
    predictions_oak = predictions_oak_df["Prediction_OAK"]


    # Plotting the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, predictions_oak, marker='o', linestyle='-', label='OAK iwth Shapley values Predictions')

    plt.xlabel('Iteration')
    plt.ylabel('Predictions')
    plt.title(f'Model Predictions over Iterations on {dataset_name} dataset')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    file_path = os.path.join(plots_dir, f"{dataset_name}_predictions.png")
    plt.savefig(file_path)
    plt.show()

