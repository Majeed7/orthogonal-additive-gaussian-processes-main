import sys
import shap
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
import os
from functions_real_data_experiments import shap_values_to_df


# Create directory for saving results
output_dir = "real_data_shap"
os.makedirs(output_dir, exist_ok=True)

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

  

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize list to store predictions
    predictions = []

    # Loop until we have at least one feature
    while X_train.shape[1] > 1:
        # Linear Regression Model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # SHAP explainers
        additive_explainer = shap.AdditiveExplainer(linear_model.predict, X_train)

        # SHAP values
        additive_shap_values = additive_explainer(X_train)

        # Calculate prediction for X_train[3]
        prediction = linear_model.predict([X_train[3]])
        predictions.append(prediction[0])

        # Reshape SHAP values to 2D array - taking the mean across the third axis
        additive_shap_values_2d = np.mean(additive_shap_values.values, axis=2) 

        # Convert SHAP values to DataFrame
        additive_df = shap_values_to_df(additive_shap_values_2d)

        # Calculate absolute values and sort indices
        abs_values = np.abs(additive_df)
        sorted_indices = np.argsort(-abs_values.values, axis=1) + 1  # adding 1 to move ranking from 0 to 1

        # Calculate average values
        avg_values = np.mean(sorted_indices, axis=0)
        print(avg_values)

        # Identify the least important feature
        least_important_feature = np.argmax(avg_values)

        # Remove the least important feature from the dataset
        X_train = np.delete(X_train, least_important_feature, axis=1)
        X_test = np.delete(X_test, least_important_feature, axis=1)

        print(f"Removed feature: {least_important_feature}")

    # Print all predictions
    print(f"Predictions for X_train[3] at each iteration for dataset {name}:", predictions)
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
    predictions_df.to_csv(f"{output_dir}/predictions_AdditiveExplainer_{name}.csv", index=False)

    print(f"Predictions saved to {output_dir}/predictions_AdditiveExplainer_{name}.csv")
    


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

  

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    predictions = []
    # Loop until we have at least one feature
    while X_train.shape[1] > 1:
        # Train a MLPRegressor model
        model = MLPRegressor(solver="lbfgs", alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
        model.fit(X_train, y_train)
        
        # Calculate prediction for X_train[3]
        prediction = model.predict([X_train[3]])
        predictions.append(prediction[0])
        
        # SHAP explainers
        X_train_summary = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(model.predict, X_train_summary)
        shap_values = explainer.shap_values(X_train)
        
        # Convert SHAP values to DataFrame
        shap_df = shap_values_to_df(shap_values)
        
        # Calculate absolute values and sort indices
        abs_values = np.abs(shap_df)
        mean_abs_values = abs_values.mean(axis=0)
        
        # Identify the least important feature
        least_important_feature = mean_abs_values.idxmin()
        least_important_index = int(least_important_feature.split('_')[1])  # Convert feature name to index
        
        # Remove the least important feature from the dataset
        X_train = np.delete(X_train, least_important_index, axis=1)
        X_test = np.delete(X_test, least_important_index, axis=1)
        
        print(f"Removed feature: {least_important_feature}")

    # Print all predictions
    print("Predictions for X_train[3] at each iteration:", predictions)

    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
    predictions_df.to_csv(f"{output_dir}/predictions_MLPRegressort_{name}.csv", index=False)
    print(f"Predictions saved to {output_dir}/predictions_MLPRegressort_{name}.csv")




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

  

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
  
    # RandomForest with Kernel Explainer
    predictions_rf = []
    while X_train.shape[1] > 1:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.ravel())

        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(shap.sample(X_train, 50))

        prediction = model.predict([X_train[3]])
        predictions_rf.append(prediction[0])

        shap_df = shap_values_to_df(np.array(shap_values))
        mean_abs_values = shap_df.abs().mean(axis=0)

        least_important_feature = mean_abs_values.idxmin()
        least_important_index = int(least_important_feature.split('_')[1])

        X_train = np.delete(X_train, least_important_index, axis=1)
        X_test = np.delete(X_test, least_important_index, axis=1)

    predictions_df = pd.DataFrame(predictions_rf, columns=["Predictions"])
    predictions_df.to_csv(f"{output_dir}/predictions_KernelExplainer_RandomForest_{name}.csv", index=False)


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

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize list to store predictions
    predictions = []

    # Loop until we have at least one feature
    while X_train.shape[1] > 1:
        # Linear Regression Model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # Exact SHAP explainers for Linear Model
        explainer = shap.LinearExplainer(linear_model, X_train)

        # SHAP values
        shap_values = explainer.shap_values(X_train)

        # Calculate prediction for X_train[3]
        prediction = linear_model.predict([X_train[3]])
        predictions.append(prediction[0])

        # Convert SHAP values to DataFrame
        shap_df = shap_values_to_df(shap_values)

        # Calculate absolute values and sort indices
        abs_values = np.abs(shap_df)
        sorted_indices = np.argsort(-abs_values.values, axis=1) + 1  # adding 1 to move ranking from 0 to 1

        # Calculate average values
        avg_values = np.mean(sorted_indices, axis=0)
        print(avg_values)

        # Identify the least important feature
        least_important_feature = np.argmax(avg_values)

        # Remove the least important feature from the dataset
        X_train = np.delete(X_train, least_important_feature, axis=1)
        X_test = np.delete(X_test, least_important_feature, axis=1)

        print(f"Removed feature: {least_important_feature}")

    # Print all predictions
    print(f"Predictions for X_train[3] at each iteration for dataset {name}:", predictions)
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
    predictions_df.to_csv(f"{output_dir}/predictions_LinearExplainer_{name}.csv", index=False)

    print(f"Predictions saved to {output_dir}/predictions_LinearExplainer_{name}.csv")


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

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize list to store predictions
    predictions_rf = []

    # Loop until we have at least one feature
    while X_train.shape[1] > 1:
        # Train a RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.ravel())

        # Exact SHAP explainers for RandomForest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Calculate prediction for X_train[3]
        prediction = model.predict([X_train[3]])
        predictions_rf.append(prediction[0])

        # Convert SHAP values to DataFrame
        shap_df = shap_values_to_df(np.array(shap_values))

        # Calculate absolute values and sort indices
        abs_values = np.abs(shap_df)
        mean_abs_values = abs_values.mean(axis=0)

        # Identify the least important feature
        least_important_feature = mean_abs_values.idxmin()
        least_important_index = int(least_important_feature.split('_')[1])  # Convert feature name to index

        # Remove the least important feature from the dataset
        X_train = np.delete(X_train, least_important_index, axis=1)
        X_test = np.delete(X_test, least_important_index, axis=1)

        print(f"Removed feature: {least_important_feature}")

    # Print all predictions
    print(f"Predictions for X_train[3] at each iteration for dataset {name}:", predictions_rf)
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions_rf, columns=["Predictions"])
    predictions_df.to_csv(f"{output_dir}/predictions_TreeExplainer_{name}.csv", index=False)

    print(f"Predictions saved to {output_dir}/predictions_TreeExplainer_{name}.csv")
