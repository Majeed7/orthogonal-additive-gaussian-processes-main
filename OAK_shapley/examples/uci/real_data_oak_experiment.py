import sys
sys.path.append("/root/OAK_shapley")
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path
import random
import torch
from tqdm import tqdm
import numpy as np
import os

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
from oak.input_measures import GaussianMeasure
from oak.ortho_rbf_kernel import OrthogonalRBFKernel
from examples.uci.functions_shap import Omega, tensorflow_to_torch, torch_to_tensorflow, numpy_to_torch
import gpflow
import os
from scipy import io
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd


# Load the CPU Performance dataset
cpu_performance = fetch_openml(name="cpu", as_frame=False)
# Load the Wine Quality dataset
wine_quality = fetch_openml(name="wine-quality-red", as_frame=False)

# Datasets
datasets = {
    "diabetes": load_diabetes(),
    "california_housing": fetch_california_housing(),
    "boston_housing": fetch_openml(data_id=531, as_frame=False),
    "cpu_performance": cpu_performance,
    "wine_quality": wine_quality
}

# Ensure the directory exists
output_dir = Path("real_data_oak")
output_dir.mkdir(exist_ok=True)

start_time = time.time()

for name, data in datasets.items():
    X, y = data.data, data.target

    # Limit the California Housing dataset for demonstration
    if name == "california_housing":
        X = X[:500]
        y = y[:500]

    if name == "wine_quality":
        X = X[:500]
        y = y[:500]

    # Shuffle data and split into training and testing sets
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Initialize OAK model
    oak = oak_model(max_interaction_depth=X.shape[1])

    # Initialise dataframe to save predictions
    predictions_df = pd.DataFrame(columns=["Iteration", "Prediction_OAK"])

    # Loop until there is one feature
    iteration = 0
    while X_train.shape[1] > 1:
        iteration += 1
        # Train the model
        oak.fit(X_train, y_train)
        N, D = X_train.shape
        base_kernel = gpflow.kernels.RBF()
        measure = GaussianMeasure(mu=0, var=1)  

        oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)
        alpha_pt = tensorflow_to_torch(oak.alpha)
        sigmas_pt = numpy_to_torch(np.array([oak.m.kernel.variances[i].numpy() for i in range(len(oak.m.kernel.variances)) if i != 0]))
        instance_shap_values = []

        # Initialize custom kernel
        orthogonal_rbf_kernel = OrthogonalRBFKernel(base_kernel, measure)
        X_train_tras = oak._transform_x(X_train)

        instance_K_per_feature= []
        for instance in X_train_tras:
            K_per_feature = np.zeros((N,D))
            for i in range(D):
                l = oak.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
                feature_column = X_train_tras[:, i].reshape(-1, 1)   # Shape (n, 1)
                instance_feature = instance[i].reshape(-1, 1)   # Shape (1, 1) 
                orthogonal_rbf_kernel.base_kernel.lengthscales = l
                K_per_feature[:,i]  = orthogonal_rbf_kernel(instance_feature, feature_column)
            instance_K_per_feature.append(K_per_feature)

        for K_per_feature in instance_K_per_feature:
            K_per_feature_pt = numpy_to_torch(K_per_feature)
            val = torch.zeros(D)
            for i in range(D):
                omega_dp, dp = Omega(K_per_feature_pt, i, sigmas_pt, q_additivity=None)
                omega_dp = omega_dp.to(torch.float64)
            
                val[i] = torch.matmul(omega_dp, alpha_pt)
            abs_values = torch.abs(val)
            sorted_indices = torch.argsort(abs_values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
            instance_shap_values.append(sorted_indices)
            
        shap_matrix = torch.stack(instance_shap_values)
        avg_values = torch.mean(shap_matrix.float(), dim=0)
        print(avg_values)
        # Predict on a specific instance and store the prediction
        specific_instance = X_train[3].reshape(1, X_train.shape[1])
        prediction = oak.predict(specific_instance)

        # Using loc to add the new row
        predictions_df.loc[len(predictions_df)] = [iteration, prediction.item()]

        # Identify and remove the feature with the highest SHAP rank
        highest_rank_feature = torch.argmax(avg_values).item()
        X_train = np.delete(X_train, highest_rank_feature, axis=1)
        X_test = np.delete(X_test, highest_rank_feature, axis=1)

        print(f"Removed feature at index {highest_rank_feature}")

    # Save predictions to a CSV file
    predictions_df.to_csv(output_dir / f"predictions_oak_{name}.csv", index=False)

# End timing
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Script executed in {elapsed_time:.2f} seconds.")
