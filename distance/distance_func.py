
import numpy as np
import torch
import torch.nn as nn
import ot


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)

def distribution_sim(down_dis, pre_dis):
    OT_dis = OT_dist(down_dis,pre_dis)
    return OT_dis
def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return l2diff(ss1, ss2)


def CMD(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        scms.append(moment_diff(sx1, sx2, i + 2))
    return sum(scms)


def OT_dist(X, X_test):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    M = torch.cdist(X, X_test, p=2.0) ** 2
    gamma = ot.emd(torch.FloatTensor(ot.unif(X.shape[0])),
                   torch.FloatTensor(ot.unif(X_test.shape[0])), M)
    loss = torch.sum(gamma * M)
    return loss


def L2_dist(x, y):
    '''
    compute the squared L2 distance between two matrics
    '''
    distx = torch.reshape(torch.sum(torch.square(x), 1), (-1, 1))
    disty = torch.reshape(torch.sum(torch.square(y), 1), (1, -1))
    dist = distx + disty
    dist -= 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
    return dist


def infonce(x, device):
    '''
    compute the squared L2 distance between two matrics
    '''

    criterion = nn.CrossEntropyLoss()

    out = torch.matmul(x, x.t()) / 0.07

    bsz = out.shape[0]
    out = out.squeeze()
    label = torch.arange(bsz).cuda(device).long()
    loss = criterion(out, label)

    return loss


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def MMD(X, Xtest):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()
    return MMD_dist


def CDAN(feature, softmax_output, ad_net, random_layer=None):
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        tmp = op_out.view(-1, softmax_output.size(1) * feature.size(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    # batch_size = softmax_output.size(0) // 2
    # dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    # return nn.BCELoss()(ad_out, dc_target)
    return ad_out
from numpy.linalg import norm

def cos_dist(A,B):
    sim = np.dot(A, B) / (norm(A) * norm(B))
    return 1-sim
