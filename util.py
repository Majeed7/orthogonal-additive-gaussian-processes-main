import numpy as np
from oak.ortho_rbf_kernel import OrthogonalRBFKernel
from oak.input_measures import GaussianMeasure
import gpflow

import tensorflow as tf

import numpy as np
import tensorflow as tf
from functools import reduce

def remove_features_one_by_one(
    shogp,
    x,                   # New input(s) for which we want predictions.
    sorted_features,     # List/array of feature indices, from least to most important.
    active_dim,          # Initial array of shape (d,) with 1/True for active dims, 0/False for inactive.
    verbose=False
):
    """
    Iteratively remove features (from least to most important) and compute predictions
    using the updated kernel each time.

    Parameters
    ----------
    shogp : object
        Your model object, containing OGP, kernel, X_base, etc.
    x : np.ndarray or tf.Tensor
        The new input(s) for which you want to compute predictions.
    sorted_features : list or np.ndarray
        Indices of features sorted by importance, with the least important first.
    active_dim : np.ndarray
        Boolean or 0/1 array indicating which features are currently active.
    verbose : bool
        If True, prints out intermediate results.

    Returns
    -------
    predictions_per_step : list
        A list of predictions after removing each feature in `sorted_features`.
        predictions_per_step[i] is the prediction vector after removing i+1 least important features.
    active_dims_per_step : list
        A list of the active_dim arrays after each removal (for reference).
    """

    # Transform the input x as in your code
    x_trans = shogp.OGP._transform_x(x).squeeze()  
    
    # Training data (already transformed, presumably)
    X_train_trans = shogp.X_base
    alpha = shogp.OGP.alpha.numpy()  # GP's coefficient vector

    # Grab the per-dimension kernel variances
    # (assuming an additive kernel with one sub-kernel per dimension)
    all_sigmas = [var.numpy() for var in shogp.OGP.m.kernel.variances]

    # Prepare containers to store predictions and active_dim states
    predictions_per_step = []
    active_dims_per_step = []

    # Ensure active_dim is boolean
    active_dim = np.array(active_dim, dtype=bool)

    for i, dim_to_remove in enumerate(sorted_features):
        # Deactivate the next least important feature
        active_dim[dim_to_remove] = False
        active_dims_per_step.append(active_dim.copy())

        # Recompute kernel with the updated active dimensions
        active_indices = np.where(active_dim)[0]

        # 1) Get the active sub-kernels
        active_kernels = [shogp.OGP.m.kernel.kernels[idx] for idx in active_indices]

        # 2) Get the corresponding variances
        sigmas_active = [all_sigmas[idx] for idx in active_indices]

        # 3) Compute each active kernel matrix for x_trans vs. X_train_trans
        kernel_matrices = [
            k(x_trans[np.newaxis, ...], X_train_trans) for k in active_kernels
        ]
        
        # 4) Combine them additively with their variances
        #    If your kernel has a helper function `compute_additive_terms`, you can use that:
        additive_terms = shogp.OGP.m.kernel.compute_additive_terms(kernel_matrices)
        aggregated_kernel = reduce(
            tf.add,
            [sigma2 * km for sigma2, km in zip(sigmas_active, additive_terms)]
        )

        # 5) Compute new prediction: aggregated_kernel @ alpha
        new_pred = aggregated_kernel @ alpha

        # Store the prediction
        predictions_per_step.append(new_pred.numpy() if hasattr(new_pred, 'numpy') else new_pred)

        # if verbose:
        #     print(f"Step {i+1}: Removed feature {dim_to_remove}, active features = {active_indices}")
        #     print(f"Prediction shape: {new_pred.shape}")

    return predictions_per_step, active_dims_per_step


def local_Shapley_values(shogp, X):
    ## Starting Shapley Value Calculation
    X_trans = shogp.OGP._transform_x(X)
    X_train_trans = shogp.X_base #self.OGP._transform_x(self.X_base)

    n, d = X_train_trans.shape
    instance_K_per_feature= []
    for instance in X_trans:
        K_per_feature = np.zeros( (n, d) )
        for i in range(d):
            l = shogp.OGP.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
            feature_column = X_train_trans[:, i].reshape(-1, 1)   # Shape (n, 1)
            instance_feature = instance[i].reshape(-1, 1)   # Shape (1, 1) 

            shogp.ORBF.base_kernel.lengthscales = l
            K_per_feature[:,i]  = shogp.ORBF(instance_feature, feature_column)
            #K_per_feature[:,i] = self.OGP.m.kernel.kernels[i](instance_feature, feature_column)

        instance_K_per_feature.append(K_per_feature)
    
    v0_scaled = np.sum(shogp.OGP.alpha) * shogp.OGP.m.kernel.variances[0].numpy()  
    sigma = shogp.OGP.scaler_y.scale_[0]
    mu = shogp.OGP.scaler_y.mean_[0]
    shogp.v0 = v0_scaled * sigma + mu

    alpha_pt = shogp.OGP.alpha.numpy()
    sigmas_pt = (np.array([shogp.OGP.m.kernel.variances[i].numpy() for i in range(len(shogp.OGP.m.kernel.variances)) if i != 0]))
    instance_shap_values = []

    for K_per_feature in instance_K_per_feature:
        K_per_feature_pt = K_per_feature
        val = np.zeros((d,))
        for i in range(d):
            omega_dp, dp = _omega(K_per_feature_pt, i, sigmas_pt,q_additivity = shogp.interaction_order)
            #omega_dp = omega_dp.to(torch.float64)
            val[i] = omega_dp @ alpha_pt

        #abs_values = np.abs(val)
        #sorted_indices = torch.argsort(abs_values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
        instance_shap_values.append(val * sigma) ## The sum of Shapley value is now sum to Y_scaled; need to return to the original space

    return np.array(instance_shap_values).squeeze()

def _omega(X, i,sigmas,q_additivity=None):

    n, d = X.shape
    if q_additivity is None:
        q_additivity = d
    
    # Reorder columns so that the i-th column is first
    idx = np.arange(d)
    idx[i] = 0
    idx[0] = i
    X = X[:, idx]

    # Initialize dp array
    dp = np.zeros((q_additivity, d, n))

    # Initial sum of features across the dataset
    sum_current = np.zeros((n,))
    
    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = X[:, j]
        sum_current += X[:, j]

    # Fill the dp table for higher orders
    for i in range(1, q_additivity):
        temp_sum = np.zeros((n,))
        for j in range(d):
            # Subtract the previous contribution of this feature when moving to the next order
            sum_current -= dp[i - 1, j, :]

            dp[i, j, :] =  (i/(i+1)) * X[:,j] * sum_current
            # dp[i, j, :] = dp[i, j, :] * (i/(i+1)) 
            temp_sum += dp[i, j, :]
        
        sum_current = temp_sum
    for i in range(q_additivity):
        dp[i,:,:] = dp[i,:,:] * sigmas[i]
    # Sum the first row of each slice
    omega = np.sum(dp[:, 0, :], axis=0)

    return omega , dp

