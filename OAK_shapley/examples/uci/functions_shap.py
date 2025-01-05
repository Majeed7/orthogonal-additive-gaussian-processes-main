import torch
import gpflow
import tensorflow as tf

def tensorflow_to_torch(tensorflow_tensor):
    numpy_array = tensorflow_tensor.numpy()
    torch_tensor = torch.from_numpy(numpy_array)
    return torch_tensor

def torch_to_tensorflow(torch_tensor):
    numpy_array = torch_tensor.numpy()
    tensorflow_tensor = tf.convert_to_tensor(numpy_array)
    return tensorflow_tensor

def numpy_to_torch(numpy_tensor):
    torch_tensor = torch.from_numpy(numpy_tensor)
    return torch_tensor


def Omega(X, i,sigmas,q_additivity=None):
    
    n, d = X.shape
    if q_additivity is None:
        q_additivity = d
    
    # Reorder columns so that the i-th column is first
    idx = torch.arange(d)
    idx[i] = 0
    idx[0] = i
    X = X[:, idx]

    # Initialize dp array
    dp = torch.zeros((q_additivity, d, n))

    # Initial sum of features across the dataset
    sum_current = torch.zeros((n,))
    
    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = X[:, j]
        sum_current += X[:, j]

    # Fill the dp table for higher orders
    for i in range(1, q_additivity):
        temp_sum = torch.zeros((n,))
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
    omega = torch.sum(dp[:, 0, :], axis=0)

    return omega , dp

def K_s_instance(X, sigmas,q_additivity=None):
        
    n, d = X.shape
    if q_additivity is None:
        q_additivity = d
    # Initialize dp array
    dp = torch.zeros((q_additivity, d, n))

    # Initial sum of features across the dataset
    sum_current = torch.zeros((n,))
    
    # Fill the first order dp (base case)
    for j in range(d):
        dp[0, j, :] = X[:, j]
        sum_current += X[:, j]

    # Fill the dp table for higher orders
    for i in range(1, q_additivity):
        temp_sum = torch.zeros((n,))
        for j in range(d):
            # Subtract the previous contribution of this feature when moving to the next order
            sum_current -= dp[i - 1, j, :]

            dp[i, j, :] =  X[:,j] * sum_current
        
            temp_sum += dp[i, j, :]
        
        sum_current = temp_sum
    for i in range(q_additivity):
        dp[i,:,:] = dp[i,:,:] * sigmas[i]
     # here i would like to some all dimentions d
    result = torch.zeros(n)
    for i in range(d):
        result += torch.sum(dp[:,i,:], axis=0)

    return result, dp