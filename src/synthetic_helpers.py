import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.uniform import Uniform
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
#sns.set()


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


def sample_meta(T, n, threshold=2., p=0.5, d=2):
    """Sample a dataset of T datasets (tasks) of n points in dimension d.
    Each task is sampled from a Gaussian distribution (unimodal) with
    probability 1-p and from a (balanced) 2-GMM with probability p (bimodal).

    Means and Variance-covariance matrices are randomly sampled.
    The paramater threshold controls the minimum distance between modes for 2-GMM.

    - var: Wishart(d,d+2,I_d)

    GENERALISE TO ARBITRARY NUMBER OF MODES !

    Args:
        T (int): number of tasks
        n (int): number of points per tasks
        threshold (float): mimimum euclidean distance between modes
        p (float): probability of drawing a bi-modal distribution
        d (int): dimension

    Returns:
        X (T,n,d), y (T,), means, covs
    """
    y = 2*Bernoulli(p).sample((T,))-1
    X = torch.zeros((T, n, d))
    p_gmm = torch.ones(2)
    means_list = []
    covs_list = []
    for t in range(T):
        if y[t] == 1:
            mean = Uniform(low=-5., high=5).sample((d,))
            th = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample()
            th /= th.norm()
            s = Uniform(low=threshold - 0.5, high=threshold + 0.5).sample()
            means = torch.stack((mean, mean + s * th))
            #covs = torch.randn(2, d, d)
            #covs = covs @ torch.transpose(covs, 1, 2) + 0.1 * torch.stack((torch.eye(d), torch.eye(d)))
            # Wishart sampling
            covs = torch.randn(2, d, d+2)
            covs = covs @ torch.transpose(covs, 1, 2)
            X[t] = GMM(means, covs, p_gmm).sample((n,))
        elif y[t] == -1:
            means = Uniform(low=-5., high=5).sample((d,))
            # covs = torch.randn(d, d)
            # covs = covs @ covs.T + 0.1 * torch.eye(d)
            covs = torch.randn(d, d+2)
            covs = covs @ covs.T
            X[t] = MultivariateNormal(means, covs).sample((n,))
        else:
            print("ERROR")

        means_list.append(means)
        covs_list.append(covs)

    return X, y, means_list, covs_list


def plot_tasks(t, X, y, means, covs, style="contour"):
    """Plot task t in dimension 1 or 2.

    Args:
        t: task index
        X: set of tasks
        y: labels
        means: set of means
        covs: set of covariances
        style: '3d' or contour plots
    """
    d = X.shape[-1]
    assert d == 1 or d == 2
    x = X[t]
    y = y[t]
    print("Label:", y)

    plt.figure(figsize=(6, 6))

    if y == 1.: # get theoretical distributions
        dist = GMM(means[t], covs[t], torch.ones(2))
    elif y == -1.:
        dist = MultivariateNormal(means[t], covs[t])

    if d == 2:
        if y == 1.: print("Distance between means:", (means[t][0] - means[t][1]).norm())
        print("Condition numbers:", torch.linalg.cond(covs[t]).T)
        x_axis = torch.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 50)
        y_axis = torch.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 50)
        X, Y = torch.meshgrid(x_axis, y_axis)
        XX = torch.stack((X.ravel(), Y.ravel())).T
        Z = torch.exp(dist.log_prob(torch.Tensor(XX)))
        Z = Z.view(X.shape).numpy()

        if style == "contour":
            CS = plt.contour(X, Y, Z, 20, cmap='plasma')
            plt.axis('equal');
            plt.scatter(x[:, 0], x[:, 1])
        elif style == "3d":
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
        else:
            print("Unknown style")

    if d == 1:
        print("Variance(s):", covs[t].T)
        x_axis = torch.linspace(x.min(), x.max(), 1000)
        density = torch.exp(dist.log_prob(x_axis.view(-1, 1)))
        plt.plot(x_axis, density.numpy())
        plt.hist(x.numpy().reshape(-1), bins=100, density=True)

    plt.show()