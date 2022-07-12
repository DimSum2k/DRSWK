import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

def GMM(means, covs, p):
    """GMM in arbitrary dimension d with K components.

    Args:
        means (Tensor): shape (K,d)
        covs (Tensor): (K,d,d), var-cov matrices
        p (Tensor): shape (K,), latent probabilities (or relative weights)

    Returns:
        MixtureSameFamily
    """
    assert p.shape[0] == means.shape[0]
    assert p.shape[0] == covs.shape[0]
    assert means.shape[1] == covs.shape[1]
    assert means.shape[1] == covs.shape[2]
    mix = Categorical(p)
    comp = MultivariateNormal(means, covs)
    gmm = MixtureSameFamily(mix, comp)
    return gmm
