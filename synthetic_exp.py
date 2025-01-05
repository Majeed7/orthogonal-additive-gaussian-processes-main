import sys

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
# import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
import os

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
import gpflow

import torch
from oak.input_measures import GaussianMeasure
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component
from oak.ortho_rbf_kernel import OrthogonalRBFKernel
from OAK_shapley.examples.uci.functions_shap import Omega, tensorflow_to_torch, torch_to_tensorflow, numpy_to_torch


matplotlib.rcParams.update({"font.size": 25})

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

expressions = [
    " 2 * np.sin(2 * X[:, 0]) + np.max(X[:, 1], 0) + X[:, 2] + np.exp(-X[:, 3])",
    "5 * np.cos(X[:, 0] + X[:, 1]) + 5 * X[:, 1] * X[:, 2] * X[:, 3] + np.exp(-X[:, 3])",
    "8 * X[:, 0] + 2 * X[:, 0] * X[:, 1] + 4 * X[:, 0] * X[:, 1] * X[:, 2] * X[:,3]",
    "5 * X[:, 0] + 3 * X[:, 1] * X[:, 2] * X[:, 3] + np.exp(-X[:, 3])"
]


# Create the folder for all models
base_save_dir = "synthetic_models"
Path(base_save_dir).mkdir(exist_ok=True)

for e in range(len(expressions)):


    # Create forlder for each expression (experiment)
    save_dir = os.path.join(base_save_dir, f"expression_{e+1}")
    Path(save_dir).mkdir(exist_ok=True)

    svr_rfe_iterations = []
    selectKBest_f_reg_iterations = []
    selectKBest_mutual_iterations = []
    lasso_iterations = []
    oak_shap_iterations = []
    for i in tqdm(range(50)): 
        X = np.random.normal(0, 1, (200, 10)) # tu zmienic z 6 na 10 
        y = eval(expressions[e])
        
        # 1. RFE (feature selection method) + SVR (model, estimator)
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=1, step=1)
        selector = selector.fit(X, y)
        svr_rfe_iterations.append(selector.ranking_)

        # 2. SelectKBest(feature selection method) + f_regression(model, scoring function)
        selector = SelectKBest(f_regression, k=4).fit(X, y)
        values = numpy_to_torch(selector.scores_)
        sorted_indices = torch.argsort(values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
        selectKBest_f_reg_iterations.append(sorted_indices)

        # 3. SelectKBest(feature selection method) + mutual_info_regression(model, scoring function)
        selector = SelectKBest(mutual_info_regression, k=4).fit(X, y)
        abs_values = torch.abs(numpy_to_torch(selector.scores_))
        sorted_indices = torch.argsort(abs_values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
        selectKBest_mutual_iterations.append(sorted_indices)


        # 4. LassoRegression (model, estimator) + take highest coefficients
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(X,y)
        abs_values = torch.abs(numpy_to_torch(lasso.coef_))
        sorted_indices = torch.argsort(abs_values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
        lasso_iterations.append(np.abs(sorted_indices))

        # 5. oak + shap
        X_train = X
        y_train = y.reshape(-1,1)
        
        oak = oak_model(max_interaction_depth=X.shape[1])
        oak.fit(X_train, y_train)
        N,D = X_train.shape
        base_kernel = gpflow.kernels.RBF()
        measure = GaussianMeasure(mu = 0, var = 1) # Example measure, adjust as needed

        # Initialize the custom kernel
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
        
        oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)
        alpha_pt = tensorflow_to_torch(oak.alpha)
        sigmas_pt = numpy_to_torch(np.array([oak.m.kernel.variances[i].numpy() for i in range(len(oak.m.kernel.variances)) if i != 0]))
        instance_shap_values = []

        for K_per_feature in instance_K_per_feature:
            K_per_feature_pt = numpy_to_torch(K_per_feature)
            val = torch.zeros(D)
            for i in range(D):
                omega_dp, dp = Omega(K_per_feature_pt, i, sigmas_pt,q_additivity=None)
                omega_dp = omega_dp.to(torch.float64)
            
                
                val[i] = torch.matmul(omega_dp, alpha_pt)
            abs_values = torch.abs(val)
            sorted_indices = torch.argsort(abs_values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
            instance_shap_values.append(sorted_indices)
        
        shap_matrix = torch.stack(instance_shap_values)
        median_values = torch.median(shap_matrix, dim=0).values
        oak_shap_iterations.append(median_values.numpy())


    torch.save(svr_rfe_iterations, os.path.join(save_dir, "svr_rfe_iter.pt"))
    torch.save(selectKBest_f_reg_iterations, os.path.join(save_dir, "slectKBest_f_reg_iter.pt"))
    torch.save(selectKBest_mutual_iterations, os.path.join(save_dir, "slectKBest_mutual_iter.pt"))
    torch.save(lasso_iterations, os.path.join(save_dir, "lasso_iter.pt"))
    torch.save(oak_shap_iterations, os.path.join(save_dir, "oak_shap_iter.pt"))




