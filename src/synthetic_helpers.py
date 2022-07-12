import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform

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

def sample_moments(T, d, K_max):
    """Sample moments of T GMMs in dimension d.
    Args:
        T (int): number of tasks
        d (int): input dimension
        K_max (int): maximum number of components
    Returns:
        y (T,), means (list), covs (list)
    """
    means_l = []
    covs_l = []
    y = Categorical(torch.ones(K_max)).sample((T,)) # uniform draw of modes
    for t in range(T):
        K = y[t]+1
        means = Uniform(low=-5., high=5.).sample((K,d))
        covs = Uniform(low=-1., high=1.).sample((K, d, d))
        a = Uniform(low=1., high=4.).sample((K,))
        B = Uniform(low=0., high=1.).sample((K, d))
        covs = a.view(-1,1,1) * covs @ torch.transpose(covs, 1, 2) + torch.stack([torch.diag(b) for b in B])
        means_l.append(means)
        covs_l.append(covs)
    y = F.one_hot(y.to(torch.int64))
    
    return means_l, covs_l, y

def sample_from_moments(means, covs, n, K_max, epoch):
    T = len(means)
    d = means[0].shape[-1]
    X = torch.zeros((T, epoch, n, d))
    for t in range(T):
        K = means[t].shape[0]
        for e in range(epoch):
            X[t,e] = GMM(means[t], covs[t], torch.ones(K)).sample((n,))
    return X


