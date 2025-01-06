from oak.input_measures import GaussianMeasure, EmpiricalMeasure
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component
from oak.ortho_rbf_kernel import OrthogonalRBFKernel
#from OAK_shapley.examples.uci.functions_shap import Omega, tensorflow_to_torch, torch_to_tensorflow, numpy_to_torch

import gpflow
import numpy as np 
import tensorflow as tf 

from oak.utils import compute_L

class SHOGP():
    def __init__(self, X, y, inte_order=None, inducing_points=None):
        n, d = X.shape
        base_kernel = gpflow.kernels.RBF()
        
        self.interaction_order = inte_order
        if inte_order == None: self.interaction_order = X.shape[1]


        if inducing_points == None:
            OGP = oak_model(max_interaction_depth=self.interaction_order) # If we want to use inducing points, add: num_inducing=20, sparse=True)
        else :
            OGP = oak_model(max_interaction_depth=self.interaction_order, num_inducing=inducing_points, sparse=True)

        OGP.fit(X, y.reshape(-1,1))
        measure = GaussianMeasure(mu = 0, var = 1) # Example measure, adjust as needed
        orthogonal_rbf_kernel = OrthogonalRBFKernel(base_kernel, measure)

        self.inducing_points = inducing_points
        self.X_train = X
        self.y_train = y
        self.X_base = OGP._transform_x(X) if inducing_points == None and OGP.sparse == False else OGP.m.inducing_variable.Z.numpy()
        self.OGP = OGP
        self.OGP.alpha = get_model_sufficient_statistics(self.OGP.m, get_L=False)
        self.ORBF = orthogonal_rbf_kernel
        self.n, self.d = n, d

    def local_Shapley_values(self, X):
        ## Starting Shapley Value Calculation
        X_trans = self.OGP._transform_x(X)
        X_train_trans = self.X_base #self.OGP._transform_x(self.X_base)

        n, d = X_train_trans.shape
        instance_K_per_feature= []
        for instance in X_trans:
            K_per_feature = np.zeros( (n, d) )
            for i in range(d):
                l = self.OGP.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
                feature_column = X_train_trans[:, i].reshape(-1, 1)   # Shape (n, 1)
                instance_feature = instance[i].reshape(-1, 1)   # Shape (1, 1) 

                self.ORBF.base_kernel.lengthscales = l
                K_per_feature[:,i]  = self.ORBF(instance_feature, feature_column)
                #K_per_feature[:,i] = self.OGP.m.kernel.kernels[i](instance_feature, feature_column)

            instance_K_per_feature.append(K_per_feature)
        
        self.v0 = np.sum(self.OGP.alpha) * self.OGP.m.kernel.variances[0].numpy()  
        
        alpha_pt = self.OGP.alpha.numpy()
        sigmas_pt = (np.array([self.OGP.m.kernel.variances[i].numpy() for i in range(len(self.OGP.m.kernel.variances)) if i != 0]))
        instance_shap_values = []

        for K_per_feature in instance_K_per_feature:
            K_per_feature_pt = K_per_feature
            val = np.zeros((d,))
            for i in range(d):
                omega_dp, dp = self._omega(K_per_feature_pt, i, sigmas_pt,q_additivity=self.interaction_order)
                #omega_dp = omega_dp.to(torch.float64)
                val[i] = omega_dp @ alpha_pt

            #abs_values = np.abs(val)
            #sorted_indices = torch.argsort(abs_values, descending=True) + 1 # adding 1 to move ranking from 0 to 1
            instance_shap_values.append(val * self.OGP.scaler_y.scale_[0]) ## The sum of Shapley value is now sum to Y_scaled; need to return to the original space

        return np.array(instance_shap_values).squeeze()
    
    def _omega(self, X, i,sigmas,q_additivity=None):
    
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
    
    '''
    Global Shapley Value Calculation
    '''
    def global_shapley_value(self):
        gamma_list = []
        for i in range(self.d):
            dim = i     
            l = self.OGP.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
            gamma = compute_L(X= self.X_base,#self.OGP.m.data[0], #self.X_train, 
                          lengthscale = l, 
                          variance = 1, # the variance of each kernel is set to one, instead we have variance for each interaction order; it is included later on in the computation of Shapley value
                          dim = dim, 
                          delta = self.ORBF.measure.var, 
                          mu = self.ORBF.measure.mu)
            gamma_list.append(gamma.astype(np.float64))

        variances = self.OGP.m.kernel.variances[1:] ## The first variance corresponds to the bias term

        ## For global explainability, in order to preserve the additive attribution, we need to keep the original data input
        # K = self.OGP.m.kernel(self.X_train)
        # Ktilde = K + np.eye(self.X_train.shape[0]) * self.OGP.m.likelihood.variance
        # L = np.linalg.cholesky(Ktilde)
        # alpha = tf.linalg.cholesky_solve(L, self.y_train.reshape(-1,1))
        alpha = self.OGP.alpha

        # gl = gamma_list
        # gl_hat = (gl[0] + gl[1] + gl[2])*(variances[0]**2) + (gl[0]*gl[1] + gl[0]*gl[2] + gl[1]*gl[2])*(variances[1]**2) + (variances[2]**2)*gl[0]*gl[1]*gl[2]
        # gl_0 = (gl[0])*(variances[0]**2) + (gl[0]*gl[1] + gl[0]*gl[2])*(variances[1]**2)*0.5 + (1/3)*(variances[2]**2)*gl[0]*gl[1]*gl[2]
        # gl_1 = (gl[1])*(variances[0]**2) + (gl[1]*gl[0] + gl[1]*gl[2])*(variances[1]**2)*0.5 + (1/3)*(variances[2]**2)*gl[0]*gl[1]*gl[2]
        # gl_2 = (gl[2])*(variances[0]**2) + (gl[2]*gl[0] + gl[2]*gl[1])*(variances[1]**2)*0.5 + (1/3)*(variances[2]**2)*gl[0]*gl[1]*gl[2]
        
        # sv = np.zeros((3,1))
        # sv[0] = tf.matmul(tf.matmul(tf.transpose(alpha), gl_0), alpha).numpy()
        # sv[1] = tf.matmul(tf.matmul(tf.transpose(alpha), gl_1), alpha).numpy()
        # sv[2] = tf.matmul(tf.matmul(tf.transpose(alpha), gl_2), alpha).numpy()

        # sum_sv =  tf.matmul(tf.matmul(tf.transpose(alpha), gl_hat), alpha).numpy()
        
        shapley_vals = np.zeros((self.d,))
        for i in range(self.d):
            gamma_hat_i, dp = self._gamma_hat(dim=i, gamma_list=gamma_list, variances = variances)
            shapley_vals[i] = tf.matmul(tf.matmul(tf.transpose(alpha), gamma_hat_i), alpha).numpy()
        
        shapley_vals = shapley_vals / (np.sum(shapley_vals) + self.OGP.m.likelihood.variance.numpy()) # normalizing Shapley value
        shapley_vals_rescaled = shapley_vals * self.OGP.scaler_y.var_

        return shapley_vals_rescaled, shapley_vals
    
    def _gamma_hat(self, dim, gamma_list, variances):
        n = gamma_list[0].shape[0]
        d = len(gamma_list)
        #if q_additivity is None:
        q_additivity = self.interaction_order
        
        # Reorder columns so that the i-th column is first
        idx = np.arange(d)
        idx[dim] = 0
        idx[0] = dim
        gamma_list = [gamma_list[i] for i in idx] 

        # Initialize dp array
        dp = np.zeros((q_additivity, d, n, n)).astype(np.float64)

        # Initial sum of features across the dataset
        sum_current = np.zeros((n,n)).astype(np.float64)
        
        # Fill the first order dp (base case)
        for j in range(d):
            dp[0, j, :, :] = gamma_list[j]
            sum_current += gamma_list[j]

        # Fill the dp table for higher orders
        for i in range(1, q_additivity):
            temp_sum = np.zeros((n,n))
            for j in range(d):
                # Subtract the previous contribution of this feature when moving to the next order
                sum_current -= dp[i - 1, j, :, :]

                dp[i, j, :] =  gamma_list[j] * sum_current
                #dp[i, j, :] =  (i/(i+1)) * X[:,j] * sum_current
                
                temp_sum += dp[i, j, :,:]
            
            sum_current = temp_sum
        
        gamma_hat = np.zeros((n,n)).astype(np.float64)
        for j in range(q_additivity):
            gamma_hat += dp[j,0,:,:] * (variances[j]**2 / (j+1))

        return gamma_hat , dp
    

    def prediction_featuresubset(self, x, active_dim):
        X_train_trans = self.X_base #self.OGP._transform_x(self.X_base)
        x_trans = self.OGP._transform_x(x[np.newaxis,:]).squeeze()
        n, d = X_train_trans.shape

        active_ind = np.where(active_dim == 1)[0]
        d_hat = len(active_dim)

        K_per_feature = np.zeros( (n, d_hat) )
        for i in active_ind:
            l = self.OGP.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
            feature_column = X_train_trans[:, i].reshape(-1, 1)   # Shape (n, 1)
            instance_feature = x_trans[i].reshape(-1, 1)   # Shape (1, 1) 

            self.ORBF.base_kernel.lengthscales = l
            K_per_feature[:,i]  = self.ORBF(instance_feature, feature_column)
            #K_per_feature[:,i] = self.OGP.m.kernel.kernels[i](instance_feature, feature_column)
        
        alpha = self.OGP.alpha.numpy()
        sigmas = (np.array([self.OGP.m.kernel.variances[i].numpy() for i in range(len(self.OGP.m.kernel.variances))]))
        
        aggregated_kernel, dp = self.kernel_aggregation_prediction(K_per_feature, sigmas[1:])
        bias = np.sum(alpha) * sigmas[0]

        transformed_pred = bias + aggregated_kernel.T @ alpha 
        return self.OGP.scaler_y.inverse_transform(transformed_pred[np.newaxis,:]) ## Test with the prediction with the model: self.OGP.predict(x[np.newaxis,:])

    def kernel_aggregation_prediction(self, k_per_feature, sigmas, q_additivity=None):
        n, d = k_per_feature.shape
        q_additivity = self.interaction_order
        
        # Initialize dp array
        dp = np.zeros((q_additivity, d, n))

        # Initial sum of features across the dataset
        sum_current = np.zeros((n,))
        
        # Fill the first order dp (base case)
        for j in range(d):
            dp[0, j, :] = k_per_feature[:, j]
            sum_current += k_per_feature[:, j]

        # Fill the dp table for higher orders
        for i in range(1, q_additivity):
            temp_sum = np.zeros((n,))
            for j in range(d):
                # Subtract the previous contribution of this feature when moving to the next order
                sum_current -= dp[i - 1, j, :]

                dp[i, j, :] =  k_per_feature[:,j] * sum_current
                # dp[i, j, :] = dp[i, j, :] * (i/(i+1)) 
                temp_sum += dp[i, j, :]
            
            sum_current = temp_sum

        for i in range(q_additivity):
            dp[i,:,:] = dp[i,:,:] * sigmas[i]
        # Sum the first row of each slice
        omega = np.sum(dp, axis=(0,1))

        return omega , dp


    def get_sobol(self):
        gamma_list = []
        for i in range(self.d):
            dim = i     
            l = self.OGP.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
            gamma = compute_L(X= self.X_base,#self.OGP.m.data[0], #self.X_train, 
                          lengthscale = l, 
                          variance = 1, # the variance of each kernel is set to one, instead we have variance for each interaction order; it is included later on in the computation of Shapley value
                          dim = dim, 
                          delta = self.ORBF.measure.var, 
                          mu = self.ORBF.measure.mu)
            gamma_list.append(gamma.astype(np.float64))

        variances = self.OGP.m.kernel.variances[1:] ## The first variance corresponds to the bias term
        alpha = self.OGP.alpha

        sobol = np.zeros((self.d,))
        for i in range(self.d):
            sobol[i] = tf.matmul(tf.matmul(tf.transpose(alpha), gamma_list[i]), alpha).numpy() * (variances[0] ** 2)

        return sobol