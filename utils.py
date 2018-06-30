import torch
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from torch.autograd import Variable


def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))

    out_affnty = affnty/col_sum
    in_affnty = np.transpose(affnty/row_sum)
    return in_affnty, out_affnty

# construct affinity matrix via supervised labels information
def labels_affnty(labels_1, labels_2):
    if (isinstance(labels_1, torch.LongTensor) or
        isinstance(labels_1, torch.Tensor)):
        labels_1 = labels_1.numpy()
    if (isinstance(labels_2, torch.LongTensor) or 
        isinstance(labels_2, torch.Tensor)):    
        labels_2 = labels_2.numpy()

    if labels_1.ndim == 1:
        affnty = np.float32(labels_2 == labels_1[:, np.newaxis])
    else:
        affnty = np.float32(np.sign(np.dot(labels_1, labels_2.T)))
    in_affnty, out_affnty = normalize(affnty)
    return torch.Tensor(in_affnty), torch.Tensor(out_affnty)

# construct affinity matrix via rbf kernel 
def rbf_affnty(X, Y, topk=10):
    X = X.numpy()
    Y = Y.numpy()

    rbf_k = rbf_kernel(X, Y)
    topk_max = np.argsort(rbf_k, axis=1)[:,-topk:]

    affnty = np.zeros(rbf_k.shape)
    for col_idx in topk_max.T:
        affnty[np.arange(rbf_k.shape[0]), col_idx] = 1.0

    in_affnty, out_affnty = normalize(affnty)
    return torch.Tensor(in_affnty), torch.Tensor(out_affnty)

def get_affnty(labels1, labels2, X, Y, topk=10):
    in_affnty1, out_affnty1 = labels_affnty(labels1, labels2)
    in_affnty2, out_affnty2 = rbf_affnty(X, Y, topk)

    in_affnty = torch.cat((in_affnty1, in_affnty2), 1)
    out_affnty = torch.cat((out_affnty1, out_affnty2), 0)
    return Variable(in_affnty).cuda(), Variable(out_affnty).cuda()
 

