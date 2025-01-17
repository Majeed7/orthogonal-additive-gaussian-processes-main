import numpy as np
from oak.ortho_rbf_kernel import OrthogonalRBFKernel
from oak.input_measures import GaussianMeasure
import gpflow

def prediction_featuresubset(shogp, x, active_dim, normalized=True):
        base_kernel = gpflow.kernels.RBF()
        measure = GaussianMeasure(mu = 0, var = 1) # Example measure, adjust as needed
        orthogonal_rbf_kernel = OrthogonalRBFKernel(base_kernel, measure)
        shogp.ORBF = orthogonal_rbf_kernel


        X_train_trans = shogp.X_base #self.OGP._transform_x(self.X_base)
        x_trans = shogp.OGP._transform_x(x[np.newaxis,:]).squeeze()
        n, d = X_train_trans.shape

        active_ind = np.where(active_dim.squeeze() == 1)[0]
        d_hat = len(active_dim)

        K_per_feature = np.zeros( (n, d_hat) )
        for i in active_ind:
            l = shogp.OGP.m.kernel.kernels[i].base_kernel.lengthscales.numpy()
            feature_column = X_train_trans[:, i].reshape(-1, 1)   # Shape (n, 1)
            instance_feature = x_trans[i].reshape(-1, 1)   # Shape (1, 1) 

            shogp.ORBF.base_kernel.lengthscales = l
            K_per_feature[:,i]  = shogp.ORBF(instance_feature, feature_column)
            #K_per_feature[:,i] = self.OGP.m.kernel.kernels[i](instance_feature, feature_column)
        
        alpha = shogp.OGP.alpha.numpy()
        sigmas = (np.array([shogp.OGP.m.kernel.variances[i].numpy() for i in range(len(shogp.OGP.m.kernel.variances))]))
        
        aggregated_kernel, dp = shogp.kernel_aggregation_prediction(K_per_feature, sigmas[1:])
        bias = np.sum(alpha) * sigmas[0]

        transformed_pred = bias + aggregated_kernel.T @ alpha 
        if normalized: return transformed_pred
        return shogp.OGP.scaler_y.inverse_transform(transformed_pred[np.newaxis,:]) ## Test with the prediction with the model: self.OGP.predict(x[np.newaxis,:])

