import torch
import pickle
from tqdm import trange
from tqdm import tqdm
from classifiers import KRR
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_multiple_kernels(Ks, K_tests, y_train, y_test, hp_clf, hp_kernel, subsample=1, save=""):
    r"""
    Args:
        Ks: list of gram matrices to train on (hp_kernel, T_train, T_Train)
        K_tests: list of gram matrices to evaluate on (hp_kernel, T_val, T_Train)
        y_train: labels train set
        y_test: labels validation set
        hp_clf: hyper-parameters for the classifier
        hp_kernel: hyper-parameters for the kernel
        subsample (int): if > 1, evaluate on subsets of the training gram matrices
        save: if True save training details
    """
    T = Ks.shape[-1]
    M = 2 if y_train.dim() == 1 else y_train.shape[1]
    assert (T / subsample)%M == 0
    assert Ks.dim() == 3
    n_Ks = len(Ks)
    assert n_Ks == len(K_tests)
    results = {}
    dict_best = {}
    print("Starting training...\n")
    res = torch.zeros((n_Ks, len(hp_clf), subsample, 2)).to(device)
    hp_clf_opt = []
    hp_kernel_opt = []
    res_opt = torch.zeros((subsample, 2)).to(device)
    for s in trange(subsample):
        stop = int((T / subsample) * (s + 1))
        assert stop%M == 0
        K_train_s, y_train_s, K_test_s = Ks[:, :stop, :stop], y_train[:stop], K_tests[:, :, :stop]
        max_acc = 0
        idx = (0, 0)
        for j in range(n_Ks):
            K, Ktest = K_train_s[j].to(device), K_test_s[j].to(device)
            for k in range(len(hp_clf)):
                clf = KRR(hp_clf[k])
                res[j, k, s] = train_and_evaluate(clf, K, Ktest, y_train_s, y_test)
                if res[j,k,s,1]>max_acc:
                    max_acc = res[j,k,s,1]
                    idx = (j,k)
                    w_opt = clf.alpha_
        tqdm.write("Acc: {}, idx: {}".format(max_acc.item(), idx))
        hp_clf_opt.append(hp_clf[idx[1]].item())
        hp_kernel_opt.append(hp_kernel[idx[0]])
        if save:
            torch.save(w_opt, save + 'weight_{}.pt'.format(stop))
        res_opt[s] = res[idx[0], idx[1], s]
    results["KRR"] = res
    dict_best["KRR"] = {"hp_clf": hp_clf_opt, "hp_kernel": hp_kernel_opt}
    if save:
        print("Saving results...\n")
        with open(save + 'full_results.pickle', 'wb') as f:
            pickle.dump(results, f)
        with open(save + 'hp_clfs.pickle', 'wb') as f:
            pickle.dump(hp_clf_opt, f)
        with open(save + 'best_hps_kernel_clf.pickle', 'wb') as f:
            pickle.dump(dict_best, f)
        torch.save(res_opt, save + 'train_val_acc.pt')

    return results, hp_clf_opt, dict_best, w_opt


def train_and_evaluate(clf, K_train, K_test, y_train, y_test):
    r"""
    Args:
        clf: classifier with a method fit and predict
        K_train (Tensor): shape (T_train,T_train)
        y_train (Tensor): shape (T_train)
        K_test (Tensor): shape (T_test,T_train)
        y_test (Tensor): shape (T_test)

    Output:
        Tensor (2): train and test accuracy
    """
    clf.fit(K_train, y_train)
    y_train_pred = clf.predict(K_train)
    y_test_pred = clf.predict(K_test)
    acc_train = (y_train_pred == y_train.argmax(dim=1)).sum() / y_train.shape[0]
    acc_test = (y_test_pred == y_test.argmax(dim=1)).sum() / y_test.shape[0]
    return torch.stack((acc_train, acc_test))


def evaluate_test(K_test, y_test, clf, path_weights, subsample):
    res = torch.zeros(subsample)
    clf.fitted = True
    for j in range(subsample):
        stop = K_test.shape[-1] // subsample * (j + 1)
        clf.alpha_ = torch.load(path_weights + "weight_{}.pt".format(stop)).to(device)
        y_test_pred = clf.predict(K_test[j,:,:stop])
        if clf.M == 2:
            assert (y_test_pred == y_test).dim() == 1
            acc_test = (y_test_pred == y_test).sum() / y_test.shape[0]
        else:
            acc_test = (y_test_pred == y_test.to(device).argmax(dim=1)).sum() / y_test.shape[0]
        res[j] = acc_test
        print(stop, res[j])
    return res

def evaluate_test_robust(K_test, y_test, clf, path_weights):
    res = torch.zeros(2)
    clf.fitted = True
    j=-2
    for stop in [950,1000]:
        clf.alpha_ = torch.load(path_weights + "weight_{}.pt".format(stop)).to(device)
        y_test_pred = clf.predict(K_test[j,:,:stop])
        if clf.M == 2:
            assert (y_test_pred == y_test).dim() == 1
            acc_test = (y_test_pred == y_test).sum() / y_test.shape[0]
        else:
            acc_test = (y_test_pred == y_test.to(device).argmax(dim=1)).sum() / y_test.shape[0]
        res[j] = acc_test
        print(stop, res[j])
        j=-1
    return res
