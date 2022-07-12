import torch
import time
from tqdm import trange
from gram_matrices import gram, gram_cross
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class k_MMD():
    def __init__(self, hp_inner, hp_outer, non_uniform=False):
        r"""
        Args:
            hp_inner: bandwidths for inner Gaussian kernel
            hp_outer: bandwidths for outer Gaussian kernel
            non_uniform: if True use non_uniform weights in (fashion)-Mnist experiments
            Do not set to True for the GMM expetiments
        """
        self.hp_inner = hp_inner
        self.hp_outer = hp_outer
        if type(hp_inner) == list:
            hp_inner = torch.tensor(hp_inner).to(device)
        self.hp_inner = hp_inner
        if type(hp_outer) == list:
            hp_outer = torch.tensor(hp_outer).to(device)
        self.hp_outer = hp_outer
        self.non_uniform = non_uniform

    def get_ps_mmd(self, x, y):
        r"""Return the scalar product between the kernel mean embeddings of empirical
        distributions x and y for all the hyperparameters in 'hp_inner'.

        Args:
            x (Tensor): shape (n,d)
            y (Tensor): shape (m,d)
        """
        (n, d), m = x.shape, y.shape[0]
        assert x.dim() == 2, "x should have shape (n,d)"
        assert y.dim() == 2, "y should have shape (m,d)"
        assert d == y.shape[1], "Inputs x and y should have the same dimension"
        n_hp = len(self.hp_inner)

        if self.non_uniform:
            c_x = x[:, -1].view(-1, 1)
            c_y = y[:, -1].view(-1, 1)
            x = x[:, :-1]
            y = y[:, :-1]
            (n, d), m = x.shape, y.shape[0]
        vect_diff = x.unsqueeze(1) - y
        assert (n, m, d) == vect_diff.shape
        if d == 1:
            vect_norm = vect_diff.squeeze() ** 2
        else:
            vect_norm = (vect_diff ** 2).sum(dim=-1)
        assert (n, m) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', self.hp_inner, vect_norm)
        assert (n_hp, n, m) == vect_g.shape
        if self.non_uniform:
            c = torch.kron(c_x, c_y.t())
            assert (n, m) == c.shape
            out = c * torch.exp(-vect_g)
            assert (n_hp, n, m) == out.shape
            ps = out.sum(dim=(-1, -2))
        else:
            ps = torch.exp(-vect_g).sum(dim=(-1, -2)).div(n * m)
        assert ps.dim() == 1
        assert ps.shape[0] == n_hp

    return ps

    def get_squared_norm_mmd(self, x):
        assert x.dim() == 2, "x should have shape (n,d)"
        return self.get_ps_mmd(x,x)

    def get_squared_dist_mmd(self, x, y):
        (n, d), m = x.shape, y.shape[0]
        assert x.dim() == 2, "x should have shape (n,d)"
        assert y.dim() == 2, "y should have shape (m,d)"
        assert d == y.shape[1], "Inputs x and y should have the same dimension"
        return self.get_squared_norm_mmd(x) + self.get_squared_norm_mmd(y) - 2*self.get_ps_mmd(x,y)

    def k(self, x, y):
        """Return the Gauss-Gauss MMD kernel evaluation between empirical
        distributions x and y for all the hyperparameters in 'hp_outer'."""
        out = torch.einsum('g, n -> gn', self.hp_outer, self.get_squared_dist_mmd(x, y))
        return torch.exp(-out)

    def k_uncoupled(self, x, y):
        assert len(self.hp_outer) == len(self.hp_inner)
        out = self.hp_outer*self.get_squared_dist_mmd(x, y)
        return torch.exp(-out)

    def get_norms_gauss_set(self, X):
        """Compute the MMD squared norms for a batch of inputs.
        Useful to get a fast computation of the outer Gram matrix"""
        X_phi = torch.zeros((len(self.hp_inner),len(X))).to(device)
        for i,x in enumerate(X):
            X_phi[:,i] = self.get_squared_norm_mmd(x.to(device))
        return X_phi

    def get_gram_gauss(self, X):
        T = len(X)
        n_hp_in = len(self.hp_inner)
        n_hp_out = len(self.hp_outer)
        X_phi = self.get_norms_gauss_set(X)
        assert (n_hp_in, T) == X_phi.shape
        K = gram(X, self.get_ps_mmd, hp=n_hp_in)
        assert (n_hp_in, T, T) == K.shape
        K = 2*K - X_phi.unsqueeze(-1) - X_phi.unsqueeze(1)
        assert (n_hp_in, T, T) == K.shape
        K = torch.einsum('l, gnm->lgnm', self.hp_outer, K)
        assert (n_hp_out, n_hp_in, T, T) == K.shape
        return torch.exp(K)

    def get_cross_gram_gauss(self, X, Y):
        T1, T2 = len(X), len(Y)
        n_hp_in = len(self.hp_inner)
        n_hp_out = len(self.hp_outer)
        X_phi = self.get_norms_gauss_set(X)
        Y_phi = self.get_norms_gauss_set(Y)
        assert (n_hp_in, T1) == X_phi.shape
        assert (n_hp_in, T2) == Y_phi.shape
        K = gram_cross(X,Y, self.get_ps_mmd, hp=n_hp_in)
        assert (n_hp_in, T2, T1) == K.shape
        K = 2*K - Y_phi.unsqueeze(-1) - X_phi.unsqueeze(1)
        assert (n_hp_in, T2, T1) == K.shape
        K = torch.einsum('l, gnm->lgnm', self.hp_outer, K)
        assert (n_hp_out, n_hp_in, T2, T1) == K.shape
        return torch.exp(K)

    def get_grams_gauss(self, X, Y):
        n_hp_in = len(self.hp_inner)
        n_hp_out = len(self.hp_outer)
        X_phi = self.get_norms_gauss_set(X)
        assert (n_hp_in, T_x) == X_phi.shape
        K = gram(X, self.get_ps_mmd, hp=n_hp_in)
        assert (n_hp_in, T_x, T_x) == K.shape
        K = 2*K - X_phi.unsqueeze(-1) - X_phi.unsqueeze(1)
        assert (n_hp_in, T_x, T_x) == K.shape
        K = torch.einsum('l, gnm->lgnm', self.hp_outer, K)
        assert (n_hp_out, n_hp_in, T_x, T_x) == K.shape
        K_xx =  torch.exp(K)
        del K

        Y_phi = self.get_norms_gauss_set(Y)
        assert (n_hp_in, T_y) == Y_phi.shape
        K = gram_cross(X,Y, self.get_ps_mmd, hp=n_hp_in)
        assert (n_hp_in, T_y, T_x) == K.shape
        K = 2*K - Y_phi.unsqueeze(-1) - X_phi.unsqueeze(1)
        assert (n_hp_in, T_y, T_x) == K.shape
        K = torch.einsum('l, gnm->lgnm', self.hp_outer, K)
        assert (n_hp_out, n_hp_in, T_y, T_x) == K.shape
        K_yx = torch.exp(K)

        return K_xx, K_yx




