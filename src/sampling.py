import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.uniform import Uniform
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import pickle


def GMM(means, covs, p):
    r"""
    Create a torch.distributions object as a GMM dimension d with K components.

    Args:
        latent probabilities (or relative weights), p (Tensor): shape (K,)
        means (Tensor) : shape (K,d)
        var-cov matrices, covs (Tensor): (K,d,d)

    Output:
        torch.distributions.mixture_same_family.MixtureSameFamily
    """

    K, d = means.shape
    mix = Categorical(p)
    comp = MultivariateNormal(means, covs)
    gmm = MixtureSameFamily(mix, comp)
    return gmm


def sample_meta(T, n, threshold=2., p=0.5, d=2, path=""):
    r"""
    Sample a dataset of T datasets (tasks) of n points in dimension d.
    Each task is sampled from a Gaussian distribution (unimodal) with
    probability 1-p and from a 2-GMM with probability p (bimodal).
    Means and variance-covariance matrices are randomly sampled.

    Args:
        T (int): number of distributions
        n (int): number of points per task
        threshold (float): distance between the means when a 2-GMM is sampled
        p (float): meta latent probability
        d (int): input dimension

    Output:
        X (Tensor): data, shape (T,n,d)
        y (Tensor): labels, shape (T,)
        means_list (list): list of means for plots
        covs_list (list): list of covariances for plots
    """

    y = 2*Bernoulli(p).sample((T,))-1 # (-1,1) labels
    X = torch.zeros((T, n, d))
    p_gmm = torch.ones(2)
    means_list = []
    covs_list = []
    print("test")

    for t in range(T):

        if y[t] == 1:
            # sample means with distance 'threshold' (with some noise)
            # mean = Uniform(low=-5., high=5).sample((d,))
            mean = torch.zeros((d,))
            th = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample()
            th /= th.norm()
            s = Uniform(low=threshold - 0.5, high=threshold + 0.5).sample()
            means = torch.stack((mean, mean + s * th))

            # sample covariance matrices as AA.T, with gaussian entries in A
            # and diagonal off-set to improve condition numbers
            covs = torch.randn(2, d, d)
            covs = covs @ torch.transpose(covs, 1, 2) + 0.1 * torch.stack((torch.eye(d), torch.eye(d)))
            X[t] = GMM(means, covs, p_gmm).sample((n,))

        elif y[t] == -1:
            #means = Uniform(low=-5., high=5).sample((d,))
            means = torch.zeros((d,))
            covs = torch.randn(d, d)
            covs = covs @ covs.T + 0.1 * torch.eye(d)
            X[t] = MultivariateNormal(means, covs).sample((n,))
        else:
            raise ValueError("CHECK LABELS")

        means_list.append(means)
        covs_list.append(covs)

    if path:
        print("Saving dataset, d={}".format(d))
        with open(path + "_data.pickle", 'wb') as f:
            pickle.dump((X,y), f)
        with open(path + "_moments.pickle", 'wb') as f:
            pickle.dump((means, covs), f)
        #path = "data/T_{}_n_{}_d_{}".format(T,n,d)
        #sample_meta(T,n,d=d,path=path)
    else:
        return X, y, means_list, covs_list


def train_test_split(X,y, train_size):
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]
