import torch
import time
import pickle
from tqdm import trange # tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gram(X, k, hp=1):
    r"""Build the Gram matrix from data matrix X and kernel k.

    Args:
        X (Tensor): data, shape (T,n,d) or X (list) list of tensors with varying size and same dimension
        k (function): kernel, take 2 Tensors (n,d) and (m,d) as inputs and outputs a float

    Output:
        G (Tensor): Gram matrix, shape (T,T)
    """

    T = len(X)
    G = torch.zeros((hp, T, T)).to(device)
    for t in trange(T - 1, desc='1st loop'):
        for s in trange(t + 1, T, desc='2nd loop', leave=False):
                G[:, t, s] = k(X[t].to(device), X[s].to(device))
    G += G.swapaxes(1,2).clone()
    for t in trange(T, desc="3rd loop"):
        G[:,t, t] = k(X[t].to(device), X[t].to(device))
    return G

def gram_cross(X_train, X_test, k,hp=1):
    r"""Build the Gram matrix between data matrices X_test and X_train from k.

    Args:
        X_train (Tensor): shape (T,n,d) (or list of tensors)
        X_test (Tensor): shape (T_test,n,d) (or list of tensors)
        k (function): kernel, take 2 Tensors (n,d) and (m,d) as inputs and outputs a float

    Output:
        G (Tensor): Gram matrix, shape (T_test,T)
    """
    print("HP", hp)
    T1, T2 = len(X_test), len(X_train)
    G = torch.zeros((hp, T1, T2)).to(device)
    for t1 in trange(T1, desc='1st loop'):
        for t2 in trange(T2, desc='2nd loop', leave=False):
                G[:, t1, t2] = k(X_test[t1].to(device), X_train[t2].to(device))
    return G

def compute_gram_matrices(X_train, X_val, k, hp, save=""):
    start_time = time.time()
    n_hp = len(hp)
    K_train = gram(X_train, k, hp=n_hp)
    print("time elapsed computing training Gram matrix: {:.2f}s (shape: {} )".format(time.time() - start_time, K_train.shape))
    start_time = time.time()
    K_val = gram_cross(X_train, X_val, k, hp=n_hp)
    print("time elapsed computing validation Gram matrix: {:.2f}s (shape: {} )\n".format(time.time() - start_time, K_val.shape))
    if save:
        torch.save(K_train, save + 'K_train.pt')
        torch.save(K_val, save + 'K_val.pt')
        torch.save(hp, save + "hp_kernel.pt")

    return K_train, K_val

def load_gram_matrices(path):
    K_train = torch.load(path + 'K_train.pt', map_location=device)
    K_val = torch.load(path + 'K_val.pt', map_location=device)
    hp = torch.load(path + 'hp_kernel.pt', map_location=device)
    print("Loading train and validation matrices...")
    print("Train gram dim: {}".format(K_train.shape))
    print("Val gram dim: {}".format(K_val.shape))
    return K_train, K_val, hp

def gram_standard_gauss(X,Y,gammas, save="", test=False):
    n,d = X.shape
    m = Y.shape[0]
    assert d == Y.shape[1]
    vect_diff = Y.unsqueeze(1) - X
    assert (m,n,d) == vect_diff.shape
    vect_norm = (vect_diff ** 2).sum(dim=-1)  # shape: m*n
    assert (m, n) == vect_norm.shape
    K_val = torch.exp(-torch.einsum('g, nm->gnm', gammas, vect_norm))
    assert (len(gammas), m, n) == K_val.shape

    if test :
        if save:
            torch.save(K_val, save + 'K_test.pt')
        else:
            return K_val

    vect_diff = X.unsqueeze(1) - X
    assert (n,n,d) == vect_diff.shape
    vect_norm = (vect_diff ** 2).sum(dim=-1)  # shape: n*n
    assert (n, n) == vect_norm.shape
    K_train = torch.exp(-torch.einsum('g, nm->gnm', gammas, vect_norm))
    assert (len(gammas), n, n) == K_train.shape

    if save:
        torch.save(K_train, save + 'K_train.pt')
        torch.save(K_val, save + 'K_val.pt')
        torch.save(gammas, save + "hp_kernel.pt")
    else:
        return K_train, K_val, gammas

def gram_hellinger(X, Y, gammas, save = "", test=False):
    n,d = X.shape
    m = Y.shape[0]
    assert d == Y.shape[1]
    # Re-normalize to get probability vectors + square root for Hellinger
    X = torch.sqrt(torch.einsum("ij,i->ij", X, 1/X.sum(1)))
    Y = torch.sqrt(torch.einsum("ij,i->ij", Y, 1/Y.sum(1)))
    vect_diff = Y.unsqueeze(1) - X
    assert (m,n,d) == vect_diff.shape
    vect_norm = (vect_diff**2).sum(dim=-1)/2
    assert (m,n) == vect_norm.shape
    K_val = torch.exp(-torch.einsum('g, nm->gnm', gammas, vect_norm)) 
    assert (len(gammas), m, n) == K_val.shape

    if test:
        if save:
            torch.save(K_val, save + 'K_test.pt')
        else:
            return K_val
    vect_diff = X.unsqueeze(1) - X
    assert (n,n,d) == vect_diff.shape
    vect_norm = (vect_diff ** 2).sum(dim=-1)/2  # shape: n*n
    assert (n, n) == vect_norm.shape
    K_train = torch.exp(-torch.einsum('g, nm->gnm', gammas, vect_norm))
    assert (len(gammas), n, n) == K_train.shape
    if save:
        torch.save(K_train, save + 'K_train.pt')
        torch.save(K_val, save + 'K_val.pt')
        torch.save(gammas, save + "hp_kernel.pt")
    else: 
        return K_train, K_val, gammas

def gram_TV(X, Y, gammas, save = "", test=False):
    n,d = X.shape
    m = Y.shape[0]
    assert d == Y.shape[1]
    # Re-normalize to get probability vectors + absolute difference for total variation
    X = torch.einsum("ij,i->ij", X, 1/X.sum(1))
    Y = torch.einsum("ij,i->ij", Y, 1/Y.sum(1))
    vect_diff = Y.unsqueeze(1) - X
    assert (m,n,d) == vect_diff.shape
    vect_abs = (torch.abs(vect_diff)).sum(dim=-1)/2
    assert (m,n) == vect_abs.shape
    K_val = torch.exp(-torch.einsum('g, nm->gnm', gammas, torch.sqrt(vect_abs)))
    assert (len(gammas), m, n) == K_val.shape

    if test:
        if save:
            torch.save(K_val, save + 'K_test.pt')
        else:
            return K_val
    vect_diff = X.unsqueeze(1) - X
    assert (n,n,d) == vect_diff.shape
    vect_abs = (torch.abs(vect_diff)).sum(dim=-1)/2  # shape: n*n
    assert (n, n) == vect_abs.shape
    K_train = torch.exp(-torch.einsum('g, nm->gnm', gammas, torch.sqrt(vect_abs)))
    assert (len(gammas), n, n) == K_train.shape
    if save:
        torch.save(K_train, save + 'K_train.pt')
        torch.save(K_val, save + 'K_val.pt')
        torch.save(gammas, save + "hp_kernel.pt")
    else: 
        return K_train, K_val, gammas
