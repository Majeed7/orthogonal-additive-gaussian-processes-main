from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target
import copy 
import shap 

import os
import pickle
import time
from openpyxl import Workbook
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression, RFECV
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from pathlib import Path
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import io 

import gpflow
from scipy.cluster.vq import kmeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from ucimlrepo import fetch_ucirepo 

from SHOGP import SHOGP
from oak.model_utils import oak_model, save_model

import dill

import warnings
warnings.filterwarnings("ignore")

import os
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data")


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train an SVM or SVR model with imputation for missing values.

    Parameters:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Testing feature matrix
        y_test: Testing labels

    Returns:
        best_model: Trained model after hyperparameter tuning
        best_params: Best hyperparameters from GridSearchCV
        score: Performance score (accuracy for classification, RMSE for regression)
    """

    # Check the type of the target variable
    target_type = type_of_target(y_train)
    is_classification = target_type in ["binary", "multiclass"]

    # Define the parameter grid
    param_grid = {
        "svm__C": [0.1, 1, 10],  # Regularization parameter
    }

    # Choose the model
    model = SVC() if is_classification else SVR()

    # Create a pipeline with an imputer and the SVM/SVR model
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
        ("svm", model)
    ])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy" if is_classification else "neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    if is_classification:
        score = accuracy_score(y_test, y_pred)
    else:
        rmse = mean_squared_error(y_test, y_pred)
        score = rmse

    print("Best Parameters:", best_params)
    print("Performance Score:", score)

    return best_model, best_params, score

def load_dataset(name):
    """
    Load dataset by name.

    Parameters:
        name: Name of the dataset

    Returns:
        X, y: Feature matrix and labels
    """
    down_sample = False
    down_sample_rate = 1 
    imbalance = False
    if name == "madelon":
        dataset = fetch_openml(name="madelon", version=1, as_frame=True)

    elif name == "nomao":
        dataset = fetch_openml(name="nomao", version=1, as_frame=True)
        down_sample=True
        imbalance = True
        down_sample_rate = 0.3
    
    elif name == "waveform":
        dataset = fetch_openml(name="waveform-5000", version=1, as_frame=True)

    elif name == "steel":
        dataset = fetch_openml(name="steel-plates-fault", version=1, as_frame=True)

    elif name == "sonar":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        dataset = pd.read_csv(url, header=None)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        return X.values, y.values
    
    elif name == "ionosphere":
        ds = io.loadmat(f'{data_dir}/ionosphere.mat')
        return ds["X"], ds["y"]

    elif name == "gas":
        data = pd.read_csv(f'{data_dir}/gas.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "breast_cancer_wisconsin3": #classification
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
        # data (as pandas dataframes) 
        X = breast_cancer_wisconsin_diagnostic.data.features 
        y = breast_cancer_wisconsin_diagnostic.data.targets 
        return X.values, y.values
    
    elif name == "breast_cancer": #regression
        data = pd.read_csv(f'{data_dir}/breastcancer.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "pol": #regression
        data = pd.read_csv(f'{data_dir}/pol.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values   
    
    elif name == "pumadyn32nm": #regression
        data = pd.read_csv(f'{data_dir}/pumadyn32nm.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values

    elif name == "skillcraft": #regression
        data = pd.read_csv(f'{data_dir}/skillcraft.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "sml": #regression
        data = pd.read_csv(f'{data_dir}/sml.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "keggdirected": #regression
        data = pd.read_csv(f'{data_dir}/keggdirected.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "keggundirected": #regression
        data = pd.read_csv(f'{data_dir}/keggundirected.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "parkinson": #regression
        data = pd.read_csv(f'{data_dir}/parkinson.csv.gz', sep=',')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X.values, y.values
    
    elif name == "crime":
        X, y = shap.datasets.communitiesandcrime()
        return X.values, y
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = dataset.data, dataset.target

    if down_sample:
        if imbalance:
            rus = RandomUnderSampler(random_state=42)
            X_downsampled, y_downsampled = rus.fit_resample(X, y)
        
        X, _, y, _ = train_test_split(X_downsampled, y_downsampled, train_size=down_sample_rate, random_state=42)

    return np.array(X), np.array(y)


# Ensure a directory exists for saving models
os.makedirs("trained_models", exist_ok=True)

# Define the list of feature selectors
feature_selectors = ["AGP-SHAP", "Sobol", "mutual_info", "lasso", "k_best", "tree_ensemble"] #, "rfecv"]

# Initialize an Excel workbook to store global importance values
wb = Workbook()

results_xsl = Path('agp_fs_real.xlsx')
if not os.path.exists(results_xsl):
    # Create an empty Excel file if it doesn't exist
    pd.DataFrame().to_excel(results_xsl, index=False)



if __name__ == '__main__':
    # steel: 1941 * 33    binary
    # waveform: 5000 * 40 binary 
    # sonar: 208 * 61 binary 
    
    # nomao: 34465 * 118 binary

    #dataset_names = ["breast_cancer", "sonar", "nomao", "waveform"] #"steel", "ionosphere", "gas", "pol", "sml"]
    #dataset_names2 = ["breast_cancer_wisconsin", "pumadyn32nm", "skillcraft", "crime"]
    dataset_names3 = ['keggdirected', 'parkinson']
    # Main running part of the script
    for dataset_name in dataset_names3:
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            X, y = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        # Determine if the dataset is for classification or regression
        mode = "classification" if type_of_target(y) in ["binary", "multiclass"] else "regression"
        if mode != "regression":
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(y)
            y = label_encoder.fit_transform(y).reshape(-1, 1)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        n, d = X_train.shape
        
        '''
        Train Support Vector Machine with RBF kernel
        '''
        # Train SVM on the full dataset and store the best model
        print("Training SVM on the full dataset...")
        best_model, best_params, full_score = train_svm(X_train, y_train, X_test, y_test)

        # Save the trained model to a file
        model_filename = f"trained_models/svm_{dataset_name}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Saved best SVM model for {dataset_name} to {model_filename}")

        # Prepare an Excel sheet for the current dataset
        sheet = wb.create_sheet(title=dataset_name)
        sheet.append(["Feature Selector", "Execution Time"] + [f"Feature {i}" for i in range(X.shape[1])])

        # Apply each feature selector
        for selector in feature_selectors:
            print(f"Applying feature selector: {selector} on dataset: {dataset_name}")
            start_time = time.time()

            if selector == "AGP-SHAP" or selector == "Sobol":
                if X.shape[1] >=60:
                    int_order = 3
                else:  
                    int_order = 5
                
                if 'shogp' not in locals():
                    start_time = time.time()
                    shogp = SHOGP(X_train, y_train, inte_order=int_order, inducing_points=200)
                    print(f"SHOGP model created in {time.time() - start_time} seconds")
                    # Save the model
                    with open(f"trained_models/shogp_{dataset_name}.pkl", "wb") as f:
                        dill.dump(shogp, f)
                    
                    # load the model    
                    #with open(f"trained_models/{dataset_name}.pkl", "rb") as f:
                    #    loaded_model = dill.load(f)
                
                if selector == "AGP-SHAP":
                    start_time = time.time()
                    shogp_values, shogp_values_scaled = shogp.global_shapley_value()
                    global_importance = (shogp_values)
                else:
                    start_time = time.time()
                    global_importance = shogp.get_sobol()

            elif selector == "mutual_info":
                global_importance = mutual_info_classif(X, y) if mode == "classification" else mutual_info_regression(X, y)

            elif selector == "lasso":
                lasso = Lasso().fit(X, y)
                global_importance = np.abs(lasso.coef_)

            elif selector == "rfecv":
                estimator = SVC(kernel="linear") if mode == "classification" else SVR(kernel="linear")
                rfecv = RFECV(estimator, step=1, cv=5)
                rfecv.fit(X, y)
                global_importance = rfecv.ranking_

            elif selector == "k_best":
                bestfeatures = SelectKBest(score_func=f_classif, k="all") if mode == "classification" else SelectKBest(score_func=f_regression, k="all")
                fit = bestfeatures.fit(X, y)
                global_importance = fit.scores_

            elif selector == "tree_ensemble":
                model = ExtraTreesClassifier(n_estimators=50) if mode == "classification" else ExtraTreesRegressor(n_estimators=50)
                model.fit(X, y)
                global_importance = model.feature_importances_

            else:
                print(f"Unknown feature selector: {selector}")
                continue

            execution_time = time.time() - start_time
            print(f"Execution time for {selector}: {execution_time}")

            # Store global importance values in the Excel sheet
            sheet.append([selector, execution_time] + list(global_importance))

        # Save the Excel file after processing each dataset
        excel_filename = "feature_importance_3.xlsx"
        wb.save(excel_filename)
        print(f"Global feature importance for {dataset_name} saved to {excel_filename}")
        del shogp
    wb.close()
    print("All datasets processed!")
    