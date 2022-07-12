import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import pickle
import os
from tqdm import trange
import gzip
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_dataset(path_raw, path_out, digits, fashion):
    """Build and save datasets from raw files

    Args:
        path_raw: path to raw data
        path_out: path to save processed data
        digits: list of classes to extract
    """
    # import data from raw file
    X_train, y_train = import_mnist_from_file(path_raw, kind='train')
    X_test, y_test = import_mnist_from_file(path_raw, kind='test')

    # convert to torch
    X_train, y_train = convert_vector_to_torch(X_train, y_train)
    X_test, y_test = convert_vector_to_torch(X_test, y_test)
    print("Train input dim : {}".format(X_train.shape), "Train label length : {}".format(y_train.shape))
    print("Test input dim : {}".format(X_test.shape), "Test label length : {}\n".format(y_test.shape))

    # extract targeted classes
    if len(digits)<10:
        print("Train set digit extraction...")
        X_train, y_train = extract_digits(digits, X_train, y_train)
        print("Test set digit extraction...")
        X_test, y_test = extract_digits(digits, X_test, y_test)

    # alternate classes for balanced subsets
    X_train, y_train = reorder_classes(X_train, y_train, digits)
    X_test, y_test = reorder_classes(X_test, y_test, digits)

    # save flatten images and labels tensors -> normalisation !
    torch.save(torch.hstack((X_train / 255,y_train)), path_out + 'flatten_train_data.pickle')
    torch.save(torch.hstack((X_test / 255, y_test)), path_out + 'flatten_test_data.pickle')

    # build 2D normalized and centered non uniform histograms with thresholding of pixel intensities
    threshold = 0
    X_train = extract_hist_list(X_train.reshape((-1, 28, 28)), threshold)
    X_test = extract_hist_list(X_test.reshape((-1, 28, 28)), threshold)
    if path_out:
        print("Saving dataset in {}\n".format(path_out))
        with open(path_out + 'train_data.pickle', 'wb') as f:
            pickle.dump((X_train, y_train), f)
        with open(path_out + 'test_data.pickle', 'wb') as f:
                pickle.dump((X_test, y_test), f)
    return

def load_train_set(opt, idx = 0):
    """Import MNIST train data in the torch.tensor format located in opt.path_data
    and perform a train/validation split from the full train set.

    Args:
        opt: config, data in opt.path_data
        plot: if True plot a few examples

    Returns:
        X_train, y_train, X_val, y_val
    """
    if opt.kernel == "standard" or opt.kernel == "Hellinger":
        X = torch.load(opt.path_data + 'flatten_train_data.pickle', map_location=torch.device(device))
        print(X.shape)
        y = X[:,-10:]
        X = X[:,:-10]
    else:
        with open(opt.path_data + 'train_data.pickle', 'rb') as f:
            X, y = pickle.load(f)
        if not opt.non_uniform:
            X = [X[i][:,:2] for i in range(len(X))]
    # train-validation split
    X_train, y_train, X_val, y_val = train_val_split(X, y, opt.T_train, opt.T_val, len(opt.classes), idx=idx)
    print("Train set: {} points, freq: {}".format(len(X_train),
                                                  [round(((y_train.argmax(dim=1) == i).sum() / len(y_train)).item(), 3) for i in
                                                   range(len(opt.classes))]))
    print("Validation set: {} points, freq: {}".format(len(X_val),
                                                  [round(((y_val.argmax(dim=1) == i).sum() / len(y_val)).item(), 3) for i in
                                                   range(len(opt.classes))]))

def load_test_set(opt, idx=0):
    """Import MNIST test data in the torch.tensor format located in opt.path_data.
    Args:
        opt: config, data in opt.path_data
    Returns:
        X_test, y_test
    """
    if opt.kernel == "standard" or opt.kernel == "Hellinger":
        X = torch.load(opt.path_data + 'flatten_test_data.pickle', map_location=torch.device(device))
        y = X[:,-10:]
        X = X[:,:-10]
        print(X.max())
        print(X.min())
    else:
        with open(opt.path_data + 'test_data.pickle', 'rb') as f:
            X, y = pickle.load(f)
        if not opt.non_uniform:
            X = [X[i][:,:2] for i in range(len(X))]
    X_test = X[opt.T_test*idx:opt.T_test*(idx+1)]
    y_test = y[opt.T_test*idx:opt.T_test*(idx+1)]
    print("Test set: {} points, freq: {}".format(len(X_test),
                                                  [round(((y_test.argmax(dim=1) == i).sum() / len(y_test)).item(), 3) for i in
                                                   range(len(opt.classes))]))

def import_mnist_from_file(path: str, kind: str='train') -> tuple:
    """Import raw MNIST (or fashion MNIST) data from `path`

    Args:
        path (str): file location
        kind: if 'train' load train set elif 'test' load test set

    Returns: images (flatten), labels
    """
    if kind == 'test':
        kind = 't10k'
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


def convert_vector_to_torch(X: list, y: list) -> tuple:
    """Convert MNIST inputs/outputs data from list to torch.tensor

    Args:
        X (list): input data, list of lists of length 784
        y (list): label data, list of labels

    Returns:
        X (torch.tensor): shape (-1,28,28)
        y (torch.tensor): shape (-1)
    """
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y


def extract_digits(digits: list, X: torch.tensor, y: torch.tensor, verbose: bool = True) -> tuple:
    """Given a list of integers from 0 to 9, extract the data corresponding to those digits

    Args:
        digits (list): list of unique digits from 0 to 9
        X: input data
        y: labels
        verbose (bool): if True print the count of the digits

    Returns:
        X (torch.tensor): shape (-1,28,28)
        y (torch.tensor): shape (-1)
    """
    if verbose:
        for d in digits:
            count = len(torch.where((y == d))[0])
            print("Occurrences digits {}: {}, frequency: {:.3f}".format(d, count, count/len(y)))
    mask = np.isin(y.cpu(), digits)
    X, y = X[mask], y[mask]
    if len(digits) == 2:
        y[y == digits[0]] = -1
        y[y == digits[1]] = 1
        y = y.view(-1,1)
    return X, y

def extract_hist_list(X: list, threshold: int = 127) -> list:
    """Transform inputs from images (28*28 grids of pixel intensities) to the coordinates
    (in the 2D positive orthant) where the pixels have an intensity greater than 'threshold'.
    Coordinates are normalized and centered to take value in [-1,1]^2.
    Points are weighted by their pixel intensity.

    Args:
        X (torch.tensor): shape (-1,28,28)
        threshold: pixel intensities under this value are set to 0

    Returns:
        list of torch.tensors with shape (-1,3) where the first dimension varies between images,
        last coordinates on the second axis contains the weights for each image
    """
    X[X <= threshold] = 0
    hist = []
    for i in trange(len(X)):
        pos = torch.nonzero(X[i]).to(device)
        c = torch.tensor([X[i][idx[0]][idx[1]] for idx in pos], dtype=torch.float).to(device)
        c /= c.sum() # weight by intensities
        pos = pos / 13.5 - 1 # normalise coordinates
        hist.append(torch.hstack([pos, c.view(-1, 1)]))
    return hist

def hist_to_im(x: torch.tensor, uniform: bool=True) -> torch.tensor:
    """Revert signal as a system of coordinates to original image with binary pixel intensities

    Args:
        x (torch.tensor): (-1,2) if uniform is True else (-1,3)

    Returns:
        image, torch.tensor (28,28)
    """
    coord = ((x[:,:2]+1) * 13.5).round().int()
    out = torch.zeros((28, 28))
    for i in range(len(x)):
        out[coord[i,0],coord[i,1]] = 255 if uniform else x[i,-1]
    return out

def reorder_classes(X,y, digits):
    r"""Reorder classes by grouping digits
    Create one-hot encoding of labels if len(digits) > 2
    or binary labels in (-1,1) if len(digits) == 2

    Args:
        X (torch.tensor): inputs
        y (torch.tensor): labels
        d (list): list of digits to extract

    Returns:
        Reordered dataset (X,y)
    """
    X_d = []
    c = len(y)
    for d in digits:
        mask = (y == d)
        X_d.append([b for a, b in zip(mask, X) if a])
        c = min(c, len(X_d[-1]))
    X_d = [X_d[i][:c] for i in range(len(digits))]
    y = y[:len(digits)*c]
    out = [None]*(len(digits)*c)
    for i in range(len(digits)):
        out[i::len(digits)] = X_d[i]
        y[i::len(digits)] = i
    if len(digits)>2:
        y = F.one_hot(y.to(torch.int64))
    elif len(digits) == 2:
        y[y == 0] = -1
    else:
        print("error")
    assert len(y) == len(out)
    out = [out[i].unsqueeze(0) for i in range(len(out))]
    out = torch.cat(out)
    return out, y

def train_val_split(X, y, T_train: int, T_val: int, digits, idx):
    r"""Create a train/val split of (X,y) with balanced classes.

    Args:
        X (torch.tensor): inputs
        y (torch.tensor): labels
        T_train (int): size of the train set
        T_val (int): size of the validation set

    Returns:
        X_train, y_train, X_val, y_val
    """
    assert T_train % digits == 0
    assert T_val % digits == 0
    assert (T_train + T_val)*(idx+1) <= len(X)
    assert idx >= 0
    X_train = X[idx*T_train:(idx+1)*T_train]
    y_train = y[idx*T_train:(idx+1)*T_train]
    if idx>0:
        X_val = X[-(idx+1)*T_val:-idx*T_val]
        y_val = y[-(idx+1)*T_val:-idx*T_val]
    else:
        X_val = X[-T_val:]
        y_val = y[-T_val:]
    return X_train, y_train, X_val, y_val

def get_rot_mat(theta):
    """ Build rotation matrix from angle.
    Args:
        theta: angle in [0,2*pi]
    Returns: rotation matrix (2,2)
    """
    #theta = torch.tensor(theta.copy())
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(im, theta):
    """Apply rotation with angle theta to image im
    Args:
        im: torch tensor
        theta: angle in [0,2*pi]
    Returns: rotated image
    """
    dtype = im.dtype
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(im.shape[0],1,1)
    grid = F.affine_grid(rot_mat, im.size(), align_corners=False).type(dtype)
    im = F.grid_sample(im.clone(), grid,  align_corners=False)
    return im

def get_contour(im):
    """Extract the coordinates of the rectangle that includes the active pixel in im
    Args:
        im: torch tensor
    Returns: tuple (l,r,u,d)
    """
    u, l = torch.nonzero(im).min(axis=0).values
    d, r = torch.nonzero(im).max(axis=0).values
    return l.item(), r.item(), u.item(), d.item()

def random_translation(patch, t_max):
    """Apply a random translation to patch into an image of shape (t_max, t_max)
    Args:
        patch: torch.tensor
        t_max: shape of the output image
    Returns: translated and padded patch
    """
    n,m = patch.shape
    if n <= t_max and m <= t_max:
        t_x_max = t_max - m # maximum translation
        t_y_max = t_max - n
        t_x = torch.randint(t_x_max, (1,)) if t_x_max>0 else 0 # select random translation
        t_y = torch.randint(t_y_max, (1,)) if t_y_max>0 else 0
        out = torch.zeros((t_max, t_max))
        out[t_y:t_y+n, t_x:t_x+m] = patch
        return out, True
    else: 
        return False, False

def pad_img(im, pad_x_l, pad_x_r, pad_y_u, pad_y_d):
    """Pad image by a given number of pixel
    Args:
        im: torch.tensor
        pad_x_l: number of left pixels to pad
        pad_x_r: number of right pixels to pad
        pad_y_u: number of up pixels to pad
        pad_y_d: number of down pixels to pad
    Returns: padded images
    """
    m = im.shape[0]
    assert m == im.shape[1]
    out = torch.zeros((m + pad_y_u + pad_y_d, m + pad_x_l + pad_x_r))
    out[pad_y_u:pad_y_u+m,pad_x_l:pad_x_l+m] = im
    return out

def apply_variation(X, max_th, pad_size=34):
    T = len(X)
    X_hist = []
    X_flat = torch.zeros((T,pad_size*pad_size))
    for t in range(T):
        while True:
            x_p = pad_img(X[t].clone(), 14, 14, 14, 14)
            th = -max_th + 2 * max_th * torch.rand(1)
            x_r = rot_img(x_p.clone().unsqueeze(0).unsqueeze(0), th).squeeze()
            l, r, u, d = get_contour(x_r)
            x_r_t, flag = random_translation(x_r[u:d, l:r], pad_size)
            if flag:
                break
        X_flat[t] = x_r_t.flatten()
        pos = torch.nonzero(x_r_t)
        c = torch.tensor([x_r_t[idx[0]][idx[1]] for idx in pos], dtype=torch.float)
        c /= c.sum()
        pos = pos / 13.5 - 1
        X_hist.append(torch.hstack([pos, c.view(-1, 1)]))
    return X_flat, X_hist

def im_variation(x, hist=True, scaled=True, th=None):
    if th is None:
        max_th = np.pi / 12
        th = -max_th + 2 * max_th * torch.rand(1)
    composed = transforms.Compose([transforms.RandomCrop(25),
                                   transforms.Resize(28)])

    if hist:
        x = hist_to_im(x.clone(), uniform=False)

    x_rot = rot_img(x.unsqueeze(0).unsqueeze(0), th)
    if scaled:
        x_rot = composed(x_rot)

    return x_rot.squeeze()
