from typing import Optional, Union
import torch
from torch import Tensor
from linear_operator import LinearOperator, to_dense
from jaxtyping import Float

import gpytorch
from torch import nn
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

def compute_interaction_terms(
    covars: Float[Union[LinearOperator, Tensor], "... D N N"],
    max_degree: Optional[int] = None,
    dim: int = -3,
) -> Float[Tensor, "max_degree N N"]:
    """
    Modified version of the GPyTorch sum_interaction_terms function.
    Instead of summing all interaction terms, it returns individual interaction terms
    for each degree.
    """
    if dim >= 0:
        raise ValueError("Argument 'dim' must be a negative integer.")

    covars = to_dense(covars)
    ks = torch.arange(max_degree, dtype=covars.dtype, device=covars.device)
    neg_one = torch.tensor(-1.0, dtype=covars.dtype, device=covars.device)

    # S_times_factor[k] = factor[k] * S[k]
    S_times_factor_ks = torch.vmap(lambda k: neg_one.pow(k) * torch.sum(covars.pow(k + 1), dim=dim))(ks)

    E_ks = torch.empty((max_degree, *S_times_factor_ks.shape[1:]), dtype=covars.dtype, device=covars.device)
    E_ks[0] = S_times_factor_ks[0]
    for deg in range(1, max_degree):
        sum_term = torch.einsum("m...,m...->...", S_times_factor_ks[:deg], E_ks[:deg].flip(0))
        E_ks[deg] = (S_times_factor_ks[deg] + sum_term) / (deg + 1)

    # Instead of summing, return the individual interaction terms for each degree
    return E_ks


class OrthogonalRBFKernel(gpytorch.kernels.Kernel):
    def __init__(self, mu, delta, **kwargs):
        super().__init__(has_lengthscale=True, **kwargs)
        self.mu = mu
        self.delta = delta

        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(1.0))  # Initial value for lengthscale
        )

        # Ensure the lengthscale is positive using a constraint
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    def forward(self, x1, x2, **params):
        # Ensure x1 and x2 are 2D tensors
        if x1.ndim == 1:
            x1 = x1.unsqueeze(-1)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(-1)

        # Squared Euclidean distance
        sq_dist = torch.cdist(x1, x2, p=2) ** 2

        # Base RBF kernel part
        base_rbf_part = torch.exp(-sq_dist / (2 * self.lengthscale ** 2))

        # Factor term
        factor = (self.lengthscale * torch.sqrt(self.lengthscale ** 2 + 2 * self.delta ** 2)) / (self.lengthscale ** 2 + self.delta ** 2)

        # Exponential term
        #exp_part = torch.exp(-((x1 - self.mu) ** 2 + (x2 - self.mu) ** 2).sum(-1) / (2 * (self.lengthscale ** 2 + self.delta ** 2)))

        x1_centered = x1 - self.mu.unsqueeze(1)  # Shape: (d, n1, 1)
        x2_centered = x2 - self.mu.unsqueeze(1)  # Shape: (d, n2, 1)

        # Compute squared differences for each feature
        sq_dist_x1 = x1_centered**2  # Shape: (d, n1, 1)
        sq_dist_x2 = x2_centered**2  # Shape: (d, n2, 1)

        # Combine the distances for every pair
        # Broadcasting to compute pairwise sums
        sq_dist_secondpart = sq_dist_x1.unsqueeze(2) + sq_dist_x2.unsqueeze(1)  # Shape: (d, n1, n2)
        exp_part = torch.exp(-sq_dist_secondpart / (2 * self.delta ** 2 + 2 * self.lengthscale ** 2)).squeeze()

        # Final kernel
        kernel_value = base_rbf_part - factor * exp_part

        return kernel_value

class AdditiveGP(ExactGP):
    def __init__(self, X_train, y_train, d, max_degree):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(X_train, y_train, likelihood)
        n, d = X_train.shape

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = OrthogonalRBFKernel(mu=torch.zeros((d,1)), delta=1, batch_shape=torch.Size([d]), ard_num_dims=1) # RBFKernel(batch_shape=torch.Size([d]), ard_num_dims=1) # 
        #ScaleKernel()

        self.max_degree = max_degree

        # Trainable variance parameters
        self.sigma_0 = nn.Parameter(torch.tensor(1.0))  # Bias term
        self.sigma_k = nn.Parameter(torch.ones(max_degree))  # Variance for each interaction degree

    def forward(self, X):
        mean = self.mean_module(X)

        # Convert input X to batched format for RBF kernel computation
        batched_dimensions_of_X = X.mT.unsqueeze(-1)
        univariate_rbf_covars = self.covar_module(batched_dimensions_of_X)

        # Compute individual interaction terms for each degree
        interaction_terms = compute_interaction_terms(
            univariate_rbf_covars, max_degree=self.max_degree, dim=-3
        )

        # Compute the weighted sum: sigma_0^2 + sum(sigma_k^2 * E_k)
        weighted_covar = self.sigma_0**2 + torch.einsum("k,k...->...", self.sigma_k**2, interaction_terms)

        return MultivariateNormal(mean, weighted_covar)


def train_gp_model(model, X_train, y_train, num_iterations=100, lr=0.01):
    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss.item():.4f}")
    print("Training complete!")


# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 50, 3
    X_train = torch.randn(n, d)
    y_train = torch.sin(X_train[:, 0]) + torch.randn(n) * 0.1

    model = AdditiveGP(X_train, y_train, d, max_degree=3)
    train_gp_model(model, X_train, y_train)

    # Test predictions
    model.eval()
    model.likelihood.eval()
    X_test = torch.randn(20, d)
    y_test = y_train = torch.sin(X_test[:, 0]) + torch.randn(20) * 0.1
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model.likelihood(model(X_test))
        print("Test predictions:", preds.mean)
