import warnings
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
# device = "cpu"



class k_sw_rf():
    def __init__(self, M=1, r=0, non_uniform=False, d_in=2, kernel="gauss", deterministic=False):
        """Sliced-Wasserstein-Kernel with Random features
        If d_in = 1, it uses the standard wasserstein kernel for compute (no slices)

        Args:
            M: number of slices
            r: number of point per slices (if r=0, compute the inner integral in closed form
                no random features)
            non_uniform: if true use the normalized pixel intensities
            d_in: dimension of the inputs
            kernel: type of kernel among ["poly", "gauss"]
            deterministic: if deterministic, use a deterministic sampling of the sphere and (0,1)

            # CHECK RF PROPERLY ! - full polynomial support
        """
        self.M = M
        self.r = r # what happen theoretically if M != r
        self.d_in = d_in
        self.kernel = kernel
        self.non_uniform = non_uniform
        # sample directions to project on
        if deterministic: # convert to torch
            if d_in == 3:
                u, v = np.mgrid[0:2 * np.pi:complex(imag=M/2), 0:np.pi:complex(imag=M/2)]
                x = np.cos(u) * np.sin(v)
                y = np.sin(u) * np.sin(v)
                z = np.cos(v)
                th = torch.tensor(np.stack((x.flatten(), y.flatten(), z.flatten()))).T
            elif d_in == 2:
                s = np.linspace(0, 2 * np.pi, M)
                th = torch.tensor(np.array([[np.cos(t), np.sin(t)] for t in s]), dtype=torch.float)
                print("Hey you")
            else:
                print("ERROR")
        else:
            th = MultivariateNormal(torch.zeros(d_in), torch.eye(d_in)).sample((M,)).to(device)
            th = F.normalize(th)  # shape: M * d
        assert (M, d_in) == th.shape
        self.th = th.to(device)
        if self.r > 0:
            # sample points on the real line to evaluate features
            if deterministic:
                self.ts = torch.linspace(0,1,r).to(device)
                print("Hey you yoo")
            else:
                self.ts = torch.rand(r).to(device)
        #print(self.th)
        #print(self.ts)
        n_ts = len(self.ts) if self.r > 0 else self.r
        print("Number of slices: {}, t's: {}".format(self.th.shape[0], n_ts))

    def _bins_indexes(self, real_numbers, bins):
        """Converts a batch of real numbers to a batch of indexes for the bins
        where the real numbers fall in.
        """
        v = (real_numbers.view(-1, 1) - bins.view(1, -1))
        v[v < 0] = float("Inf")
        # _, indexes = (real_numbers.view(-1, 1) - bins.view(1, -1)).abs().min(dim=1)
        return v.min(dim=1)[1]

    def _bins_indexes_vect(self, real_numbers, bins):
        """Batch vectorised version of bins_indexes"""
        a = real_numbers.unsqueeze(-1)
        b = bins.t().unsqueeze(0).unsqueeze(2)
        v = (a - b).squeeze(0).T
        v[v < 0] = float("Inf")
        return v.min(dim=-1)[1]

    def get_features(self,x):
        """Compute phi(x)"""
        if self.non_uniform:
            c = x[:,-1]
            x = x[:,:-1]
        n = x.shape[0]
        x_proj, indexes = (self.th @ x.T).sort(dim=-1) # shape: M * n
        assert (self.M, n) == x_proj.shape
        if self.non_uniform:
            c = c[indexes] # reorder weights
            assert (self.M, n) == c.shape
            bins = torch.hstack((torch.zeros((c.shape[0],1)).to(device), torch.cumsum(c, dim=1)[:,:-1])).to(device)
            idx_x = self._bins_indexes_vect(self.ts, bins)
            temp = [x_proj[i,idx_x[i]].view(1,-1) for i in range(self.M)]
            phi_x = torch.vstack(temp).flatten()
        else:
            #print("Hell yeah")
            bins = torch.arange(n).to(device) / n
            idx_x = self._bins_indexes(self.ts, bins)
            phi_x = x_proj[:, idx_x].flatten()
        assert self.r * self.M == len(phi_x)
        return phi_x

    def get_features_set(self, X):
        """Compute the features for a batch of inputs """
        X_phi = torch.zeros((len(X), self.r * self.M)).to(device)
        for i,x in enumerate(X):
            X_phi[i] = self.get_features(x.to(device))
        return X_phi

    def get_grams(self, X, Y, gammas=1., degrees=1):
        """Compute both K(X,Y) and K(X,X) with low computation. ADD POLY
        """
        T_x, T_y = len(X), len(Y)
        C = self.r * self.M
        g = len(gammas)

        X_phi = self.get_features_set(X)
        vect_diff = X_phi.unsqueeze(1) - X_phi
        assert (T_x, T_x, C) == vect_diff.shape
        vect_norm = (vect_diff ** 2).sum(dim=-1)
        assert (T_x, T_x) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T_x, T_x) == vect_g.shape
        K_xx = torch.exp(-vect_g / self.r / self.M)
        del vect_diff
        del vect_norm
        del vect_g

        Y_phi = self.get_features_set(Y)
        print(X_phi.shape, Y_phi.shape)
        vect_diff = Y_phi.unsqueeze(1) - X_phi
        assert (T_y, T_x, C) == vect_diff.shape
        vect_norm = (vect_diff ** 2).sum(dim=-1)
        assert (T_y, T_x) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T_y, T_x) == vect_g.shape
        K_yx = torch.exp(-vect_g / self.r / self.M )

        return K_xx, K_yx


    def get_gram(self, X, gammas=1., degrees=1):
        """Compute the gram matrix between for X
        #### ADD POLY ###
        """
        T = len(X)
        C = self.r * self.M
        g = len(gammas)
        X_phi = self.get_features_set(X)
        vect_diff = X_phi.unsqueeze(1) - X_phi
        assert (T, T, C) == vect_diff.shape
        vect_norm = (vect_diff ** 2).sum(dim=-1)
        assert (T, T) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T, T) == vect_g.shape
        K = torch.exp(-vect_g / self.r / self.M)
        return K

    def get_cross_gram(self, X, Y, gammas=1., degrees=1):
        """Compute the cross gram matrix between X and Y, K(Y,X).
        #### ADD POLY ###
        """
        T1, T2 = len(X), len(Y)
        C = self.r * self.M
        g = len(gammas)
        X_phi = self.get_features_set(X)
        Y_phi = self.get_features_set(Y)
        print(X_phi.shape, Y_phi.shape)
        vect_diff = Y_phi.unsqueeze(1) - X_phi
        assert (T2, T1, C) == vect_diff.shape
        vect_norm = (vect_diff ** 2).sum(dim=-1)
        assert (T2, T1) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T2, T1) == vect_g.shape
        K = torch.exp(-vect_g / self.r / self.M )
        return K

    def compute(self, x, y, gammas=1., degrees=1):
        """Return the value of the kernel between x and y.

        Args:
            x (Tensor): shape (n,d)
            y (Tensor): shape (m,d)
            gammas (Tensor or float): length-scale for the Gaussian kernel
            degrees (int): degree for the polynomial kernel
            ### add multiple degrees for polynomial! ###

        Returns:
            k(x,y)
        """
        n, d = x.shape
        assert x.dim() == 2, "x should have shape (n,d)"
        assert y.dim() == 2, "y should have shape (m,d)"
        m = y.shape[0]
        assert d == y.shape[1], "Inputs x and y should have the same dimension"
        assert d == self.d_in + self.non_uniform
        if d == 1:
            print("You should not bet here")
            warnings.warn("Inputs have dimension 1")
            #warnings.warn("Inputs have dimension 1, returning k_w instead")
            #return k_w(x, y, kernel=self.kernel, gamma=self.gammas)
            return
        if self.r == 0:
            return self._compute_no_rf(x, y, gammas=gammas, degrees=degrees)


        # project and sort (no need for hack)
        phi_x = self.get_features(x)
        phi_y = self.get_features(y)

        if self.kernel == "poly":
            # non-homogeneous parameter ?
            out = (phi_x.dot(phi_y)/self.r/self.M)**degrees
        elif self.kernel == "gauss":
            out = (phi_x - phi_y) ** 2
            assert self.r * self.M == len(out)
            out = torch.exp(-out.sum()/self.r/self.M*gammas)
        return out

    def _compute_no_rf(self, x, y, gammas, degrees=0):
        """Computation of the kernel without random features
        (closed form computation of the inner integral)
            ### add multiple degrees for polynomial! ###
        """
        # project and sort
        n,m = x.shape[0], y.shape[0]
        x_proj = (self.th @ x.T).sort(dim=-1)[0]  # shape: M * n
        y_proj = (self.th @ y.T).sort(dim=-1)[0]  # shape: M * m
        assert (self.M, n) == x_proj.shape
        assert (self.M, m) == y_proj.shape
        if n != m:
            n = n * m
            x_proj, y_proj = compute_hack(x_proj.unsqueeze(-1), y_proj.unsqueeze(-1))
            assert (self.M, n) == x_proj.shape
            assert (self.M, n) == y_proj.shape
        if self.kernel == "linear":
            out = x_proj * y_proj  # component-wise multiplication
            assert (self.M, n) == out.shape
            out = out.sum() / self.M / n
        elif self.kernel == "gauss":
            out = (x_proj - y_proj) ** 2  # component-wise difference
            assert (self.M, n) == out.shape
            out = out.sum() / self.M / n
            # vect_g = torch.einsum('g, nm->gnm', gammas, out)
            # out = torch.exp(-vect_g)
            # print(gammas)
            # print(out)
            out = torch.exp(-out * gammas)
        else:
            raise ValueError("Unknown kernel")
        return out





def k_w(x, y, kernel="poly", gamma=1., degree=1):
    r"""Wasserstein-Kernel.

    Args:
        x (torch.Tensor): shape (n,1)
        y (torch.Tensor): shape (m,1)
        kernel (str) : "linear" or "gauss"
        gamma (float): width of the gaussian kernel

    Output:
        k(x,y) (float)
    """

    n, m = x.shape[0], y.shape[0]
    assert x.dim() == 2, "x should have shape (n,1)"
    assert y.dim() == 2, "y should have shape (m,1)"
    assert x.shape[1] == 1, "x should dimension 1"
    assert y.shape[1] == 1, "y should dimension 1"

    if n == m:
        return k_w_balanced(x, y, kernel=kernel, gamma=gamma, degree=degree)
    else:
        return k_w_unbalanced(x, y, kernel=kernel, gamma=gamma, degree=degree)


def k_w_balanced(x, y, kernel="poly", gamma=1., degree=1):
    r"""Wasserstein-Kernel. Balanced version

    Args:
        x,y (torch.Tensor): shape (n,1)
        kernel (str) : "linear" or "gauss"
        gamma (float): width of the gaussian kernel

    Output:
        k(x,y) (float)
    """

    n, m = x.shape[0], y.shape[0]
    assert n == m, "Inputs x and y should have the same length"
    assert x.dim() == 2, "x should have shape (n,1)"
    assert y.dim() == 2, "y should have shape (n,1)"
    assert x.shape[1] == 1, "x should dimension 1"
    assert y.shape[1] == 1, "y should dimension 1"

    x_sorted = x.sort(dim=0)[0]
    y_sorted = y.sort(dim=0)[0]
    assert (n, 1) == x_sorted.shape
    assert (n, 1) == y_sorted.shape

    if kernel == "poly":
        out = (x_sorted.T @ y_sorted / n)**degree
    elif kernel == "gauss":
        out = (x_sorted - y_sorted) ** 2
        assert (n, 1) == out.shape
        out = torch.exp(-gamma / n * out.sum())
    else:
        raise ValueError("Unknown kernel")

    return out#.item()


def k_w_unbalanced(x, y, kernel="linear", gamma=1., sorted=False):
    r"""Wasserstein-Kernel. Unbalanced version

    Args:
        x (torch.Tensor): shape (n,1)
        y (torch.Tensor): shape (m,1)
        kernel (str) : "linear" or "gauss"
        gamma (float): width of the gaussian kernel

    Output:
        k(x,y) (float)
    """

    n, m = x.shape[0], y.shape[0]
    assert x.dim() == 2, "x should have shape (n,1)"
    assert y.dim() == 2, "y should have shape (m,1)"
    assert x.shape[1] == 1, "x should dimension 1"
    assert y.shape[1] == 1, "y should dimension 1"

    # bins_x = [k / n for k in range(n+1)]
    # bins_y = [k / m for k in range(m + 1)]
    # bins_inter = torch.unique(torch.tensor(bins_x + bins_y))

    if not sorted:
        x = x.sort(dim=0)[0]
        y = y.sort(dim=0)[0]
        assert (n, 1) == x.shape
        assert (m, 1) == y.shape

    x_idx, y_idx = 1, 1
    u = 0
    k = 0
    while (x_idx <= n) or (y_idx <= m):
        bool_test = (m * x_idx < n * y_idx)
        bool_dupl = (m * x_idx == n * y_idx)
        u_temp = u
        u = x_idx / n if bool_test else y_idx / m
        if kernel == "linear":
            k += x[x_idx - 1] * y[y_idx - 1] * (u - u_temp)
        elif kernel == "gauss":
            k += ((x[x_idx - 1] - y[y_idx - 1]) ** 2) * (u - u_temp)
        else:
            raise ValueError("Unknown kernel")
        # conditions to increase the idx
        x_idx += 1 if bool_test else bool_dupl
        y_idx += 1 if not bool_test else bool_dupl

    if kernel == "linear":
        return k
    else:
        return torch.exp(-gamma * k)


def compute_hack(x, y):
    # print("hack")
    n, m = x.shape[1], y.shape[1]
    t_m = torch.ones((m,)).view(-1, 1).to(device)
    t_n = torch.ones((n,)).view(-1, 1).to(device)
    x_ext = (x @ t_m.T).flatten(start_dim=1, end_dim=2)
    y_ext = (y @ t_n.T).flatten(start_dim=1, end_dim=2)
    # print(x_ext.shape, y_ext.shape)
    return x_ext, y_ext


def compute_hack_bis(x, y):
    n, m = x.shape[1], y.shape[1]
    t_n = torch.ones((n, 1))
    t_m = torch.ones((m, 1))
    x_ext = x.kron(t_m.t())
    y_ext = y.kron(t_n.t())
    return x_ext, y_ext

def k_MMD_poly_lin(x, y, degrees):
    r"""MMD kernel with:
        - linear outer kernel
        - polynomial inner kernel of degree r

    Args:
        x (Tensor): shape (n,d)
        y (Tensor): shape (m,d)
        degrees (list): degrees of the polynomial kernel

    Output:
        k(x,y) (float)
    """
    (n, d), m = x.shape, y.shape[0]
    assert x.dim() == 2, "x should have shape (n,d)"
    assert y.dim() == 2, "y should have shape (m,d)"
    assert d == y.shape[1], "Inputs x and y should have the same dimension"
    r = len(degrees)
    if type(degrees) == list:
        degrees = torch.tensor(degrees)
    out = (x @ y.T + 1)
    assert (n, m) == out.shape
    out = out.unsqueeze(-1)
    vect_r = out ** degrees
    out = vect_r.swapaxes(0,2).swapaxes(1,2)
    assert (r, n, m) == out.shape
    out = out.sum(dim=(-1, -2)).div(n * m)
    return out

def k_MMD_gauss_poly(x,y, d=10, gammas = [1.], non_uniform=False):
    ps = k_MMD_gauss_lin(x, y, gammas=gammas, non_uniform=non_uniform)
    return ps**d

def k_MMD_gauss_lin(x, y, gammas=[1.], non_uniform=False):
    r"""MMD kernel with:
        - linear outer kernel
        - gaussian inner kernel

    Args:
        x (Tensor): shape (n,d)
        y (Tensor): shape (m,d)
        gammas (list: float): parameter of the gaussian kernel, if multiple
        parameters are given, the gram matrix is computed for each value.

    Output:
        k(x,y) (float)
    """
    if non_uniform:
        c_x = x[:,-1].view(-1,1)
        c_y = y[:, -1].view(-1,1)
        x = x[:, :-1]
        y = y[:, :-1]
    (n, d), m = x.shape, y.shape[0]
    g = len(gammas)
    if type(gammas) == list:
        gammas = torch.tensor(gammas)
    assert x.dim() == 2, "x should have shape (n,d)"
    assert y.dim() == 2, "y should have shape (m,d)"
    assert d == y.shape[1], "Inputs x and y should have the same dimension"

    # vect_diff = x[:, None] - y[None, :]  # shape: n * m * d
    vect_diff = x.unsqueeze(1) - y
    assert (n, m, d) == vect_diff.shape

    if d == 1:
        vect_norm = vect_diff.squeeze() ** 2
    else:
        # vect_diff = vect_diff.norm(dim=-1) ** 2 # DO NOT USE THIS FUNCTION
        vect_norm = (vect_diff ** 2).sum(dim=-1)  # shape: n * m
    assert (n, m) == vect_norm.shape

    vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
    assert (g, n, m) == vect_g.shape

    if non_uniform:
        c = torch.kron(c_x, c_y.t())
        assert (n,m) == c.shape
        out = c * torch.exp(-vect_g)
        assert (g, n, m) == out.shape
        out = out.sum(dim=(-1, -2))
    else:
        out = torch.exp(-vect_g).sum(dim=(-1, -2)).div(n * m)
    assert out.dim()==1
    assert out.shape[0]==g
    return out


def gram_MMD_gauss_lin(X, Y, gammas=[1.], method=True):
    r"""Compute the Gram matrix at the meta level for the Gaussian MMD Kernel

    Args:
        X (Tensor): shape (T,n,d)
        Y (Tensor): shape (R,m,d)
        gammas (list: float): parameter of the gaussian kernel, if multiple
        parameters are given, the gram matrix is computed for each value.

    Output:
        G (Tensor): shape (gammas, T, R)

    WARNING: for inputs with different length, use k_MMD_gauss_lin + gram / gram_cross
    """

    T, n, d = X.shape
    R, m, _ = Y.shape
    assert d == Y.shape[-1], "Inputs should have the same dimension"

    if method:
        start_time = time.time()
        diff = X.unsqueeze(1).unsqueeze(-2) - Y.unsqueeze(0).unsqueeze(2)  # T * R * n * m * d
        print("time elapsed 1: {:.2f}s".format(time.time() - start_time))
    else:
        start_time = time.time()
        # reshapes X to (T, R, n, m, d)
        X_rs = X.view(T, 1, n, 1, d).expand(-1, R, -1, m, -1)
        # reshapes Y to (T, R, n, m, d)
        Y_rs = Y.view(1, R, 1, m, d).expand(T, -1, n, -1, -1)
        # Compute the coefficient-wise difference
        diff = X_rs - Y_rs
        print("time elapsed 1: {:.2f}s".format(time.time() - start_time))
    assert (T, R, n, m, d) == diff.shape

    start_time = time.time()
    if d == 1:
        diff_norm = diff.squeeze(-1) ** 2
    else:
        # diff_norm = diff.norm(dim=-1) ** 2  # T * T' * n * m
        diff_norm = (diff ** 2).sum(dim=-1)
    print("time elapsed 2: {:.2f}s".format(time.time() - start_time))
    assert (T, R, n, m) == diff_norm.shape

    start_time = time.time()
    for gamma in gammas:
        diff_norm = torch.exp(-gamma * diff_norm)
    print("time elapsed 3: {:.2f}s".format(time.time() - start_time))

    start_time = time.time()
    G = diff_norm.sum(dim=(-1, -2)).div(n * m)
    print("time elapsed 4: {:.2f}s".format(time.time() - start_time))

    return G


def gram_MMD_poly_lin(X, Y, r=2):
    r"""Compute the Gram matrix at the meta level for the Polynomial MMD Kernel

    Args:
        X (Tensor): shape (T,n,d)
        Y (Tensor): shape (R,m,d)
        r (int): degree of the polynomial kernel

    Output:
        G (Tensor): shape (T,R)
    """

    T, n, d = X.shape
    R, m, _ = Y.shape
    assert d == Y.shape[-1], "Inputs should have the same dimension"

    start_time = time.time()
    # reshapes X to (T, R, n, m, d)
    X_rs = X.view(T, 1, n, 1, d).expand(-1, R, -1, m, -1)
    # reshapes Y to (T, R, n, m, d)
    Y_rs = Y.view(1, R, 1, m, d).expand(T, -1, n, -1, -1)
    # Compute the coefficient-wise difference
    pairwise_prod = X_rs * Y_rs
    print("time elapsed 1: {:.2f}s".format(time.time() - start_time))
    assert (T, R, n, m, d) == pairwise_prod.shape

    start_time = time.time()
    # if d==1
    pairwise_poly = pairwise_prod.sum(dim=-1).add(1) ** r  # T * T' * n * m
    print("time elapsed 2: {:.2f}s".format(time.time() - start_time))
    assert (T, R, n, m) == pairwise_poly.shape

    start_time = time.time()
    G = pairwise_poly.sum(dim=(-1, -2)).div(n * m)
    print("time elapsed 4: {:.2f}s".format(time.time() - start_time))

    return G
