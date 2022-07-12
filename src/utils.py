# Helper functions to deal with cuda and double precision
import torch

cuda = torch.cuda.is_available()
double_precision = False
default_tensor_str = 'torch.cuda' if cuda else 'torch'
default_tensor_str += '.DoubleTensor' if double_precision else '.FloatTensor'
torch.set_default_tensor_type(default_tensor_str)
def frnp(x):
    t = torch.from_numpy(x).cuda() if cuda else torch.from_numpy(x)
    return t if double_precision else t.float()
def tonp(x, cuda=cuda):
    return x.detach().cpu().numpy() if cuda else x.detach().numpy()