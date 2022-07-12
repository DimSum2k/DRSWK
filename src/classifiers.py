import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn import svm

class KRR():
    r"""Kernel Ridge Regression

    Args:
        lambd (float): regularisation
    """
    name = "KRR"

    def __init__(self, lambd=1.):
        self.alpha_ = None
        self.lambd = lambd
        self.fitted = False

    def fit(self, Gram_K, y):
        T = Gram_K.shape[0]
        assert T == len(y)
        self.M = y.shape[1]
        weight = torch.linalg.solve(Gram_K + T * self.lambd * torch.eye(T).to(device), y.float())
        self.alpha_ = weight
        self.fitted = True

    def predict(self, Gram_pred):
        if self.fitted:
            pred = Gram_pred @ self.alpha_
            pred = pred.argmax(dim=1)
            assert pred.dim() == 1
            return pred#.to(device)
        else:
            print("warning: fit first.")

