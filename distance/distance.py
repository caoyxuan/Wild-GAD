import numpy as np
from distance.distance_func import OT_dist, L2_dist,MMD,l2diff,distribution_sim
import os
import csv
import  torch
from math import acos as arccosine
from tqdm import tqdm
def cal_uniformity(total_emb,avg_embedding,K=10):
    diff_emb = total_emb - avg_embedding
    l2_distances = [l2diff(total_emb[i], avg_embedding) for i in range(len(total_emb))]
    total_length = total_emb.shape[0]
    start,end =0,0
    groups = []
    for i in range(K):
        start = end
        end = start + int(total_length/K)
        if i == K-1:
            end = total_length
        indices = np.argsort(l2_distances)[start:end]
        groups.append(diff_emb[indices])
    uniformities = torch.tensor(
        [uniformity_loss(torch.tensor(groups[i])) for i in range( K )])
    return uniformities
def uniformity_loss(features,t=1,max_size=30000,batch=10000):
    # calculate loss
    n = features.size(0)
    features = torch.nn.functional.normalize(features)
    if n < max_size:
        # loss = torch.pdist(features, p=2).pow(2).mul(-t).exp().mean().log()
        loss = torch.log(torch.exp(2.*t*((features@features.T)-1.)).mean())
        # print("2",loss)
    else:
        total_loss = 0.
        permutation = torch.randperm(n)
        features = features[permutation]
        for i in range(0, n, batch):
            batch_features = features[i:i + batch]
            batch_loss = torch.log(torch.exp(2.*t*((batch_features@batch_features.T)-1.)).mean())
            total_loss += batch_loss
        loss = total_loss / (n // batch)
    return loss
def r_distirbution_fit(total_emb,avg_embedding):
    l2_distances = [[l2diff(total_emb[i], avg_embedding)] for i in range(len(total_emb))]
    return l2_distances
def theta_distribution_fit3(total_emb,avg_embedding):
    theta = (total_emb-avg_embedding)    / (np.linalg.norm(total_emb-avg_embedding,axis=1)[:,None])
    return theta
def theta_distirbution_fit(total_emb,avg_embedding):
    theta =[[np.dot(total_emb[i],avg_embedding) / \
    (np.linalg.norm(total_emb[i]) * np.linalg.norm(avg_embedding))] for i in range(len(total_emb))]
    return theta
def theta_distirbution_fit2(total_emb,avg_embedding,ref_vec = 1):
    if ref_vec == 1:
        ref_vec = np.ones_like(avg_embedding)
    else:
        ref_vec = np.ones_like(avg_embedding)
        ref_vec[0] = 0
    theta = np.dot((total_emb-avg_embedding),ref_vec).reshape(-1,1)/((np.linalg.norm(total_emb-avg_embedding,axis=1)[:,None])*np.linalg.norm(ref_vec))
    # theta =[(total_emb[i]-avg_embedding) / \
    # (np.linalg.norm(total_emb[i]-avg_embedding))for i in range(len(total_emb))]
    theta = [[arccosine(theta[i])] for i in range(len(theta))]
    return theta



similarities = []
uniformities = []
results = []
def cal_distance_labeled(downstream_emb,pre_emb,pre_y):
    threshold = 0.9
    pre_y = pre_y.cpu().numpy()
    indices = np.where(pre_y == 0)[0]
    # print(indices.max(), len(pre_emb))
    indices = np.where(pre_y == 1)[0]
    # print(indices.max(), len(pre_emb))

    pre_normal_emb  = pre_emb[np.where(pre_y==0)[0]]
    # print(pre_normal_emb.shape)

    pre_abnormal_emb = pre_emb[np.where(pre_y==1)[0]]
    # print(pre_normal_emb.shape,pre_abnormal_emb.shape)
    total_emb = np.concatenate((downstream_emb, pre_emb))
    avg_before = np.mean(downstream_emb,0)
    avg_after = np.mean(total_emb,0)
    similarity_center =l2diff(torch.tensor(avg_after),torch.tensor(avg_before))
    puesdo_normal_emb = sorted(downstream_emb, key=lambda x: l2diff(torch.tensor(x), torch.tensor(avg_before)))[
                        :int(len(downstream_emb) * threshold)]
    puesdo_abnormal_emb = sorted(downstream_emb, key=lambda x: l2diff(torch.tensor(x), torch.tensor(avg_before)))
    total_normal_emb = np.concatenate((puesdo_normal_emb, pre_normal_emb))
    r_distribution_normal_before = r_distirbution_fit(torch.tensor(puesdo_normal_emb), torch.tensor(avg_before))
    r_distribution_normal_after = r_distirbution_fit(torch.tensor(total_normal_emb), torch.tensor(avg_before))
    theta_distribution_normal_before = theta_distribution_fit3(puesdo_normal_emb, avg_before)
    theta_distribution_normal_after = theta_distribution_fit3(total_normal_emb, avg_before)
    r_theta_normal_before = np.concatenate((r_distribution_normal_before, theta_distribution_normal_before), axis=1)
    r_theta_normal_after = np.concatenate((r_distribution_normal_after, theta_distribution_normal_after), axis=1)
    similarity_r_theta_normal = distribution_sim(torch.tensor(r_theta_normal_before),torch.tensor(r_theta_normal_after))
    uniformity_normal = cal_uniformity(torch.tensor(total_normal_emb),torch.tensor(avg_before)).mean()
    if pre_abnormal_emb is not None:
        total_abnormal_emb = np.concatenate((puesdo_abnormal_emb, pre_abnormal_emb))
        r_distribution_abnormal_before = r_distirbution_fit(torch.tensor(puesdo_abnormal_emb),torch.tensor(avg_before))
        r_distribution_abnormal_after = r_distirbution_fit(torch.tensor(total_abnormal_emb),torch.tensor(avg_before))
        theta_distribution_abnormal_before = theta_distirbution_fit2(puesdo_abnormal_emb,avg_before)
        theta_distribution_abnormal_after = theta_distirbution_fit2(total_abnormal_emb,avg_before)
        r_theta_abnormal_before = np.concatenate((r_distribution_abnormal_before,theta_distribution_abnormal_before),axis=1)
        r_theta_abnormal_after = np.concatenate((r_distribution_abnormal_after,theta_distribution_abnormal_after),axis=1)
        # cretiria 3
        similarity_r_theta_abnormal = distribution_sim(torch.tensor(r_theta_abnormal_before),torch.tensor(r_theta_abnormal_after))
        uniformity_abnormal = cal_uniformity(torch.tensor(total_abnormal_emb),torch.tensor(avg_before)).mean()
    else:
        similarity_r_theta_abnormal = -1e5
        uniformity_abnormal = -1e5
    return similarity_center.item(),similarity_r_theta_normal.item(),uniformity_normal.item(),similarity_r_theta_abnormal,uniformity_abnormal


def cal_distance_unlabeled(downstream_emb,pre_emb):
    threshold = 0.9
    total_emb = np.concatenate((downstream_emb, pre_emb))
    avg_before = np.mean(downstream_emb,0)
    avg_after = np.mean(total_emb,0)
    puesdo_normal_emb = sorted(downstream_emb, key=lambda x: l2diff(torch.tensor(x), torch.tensor(avg_before)))[
                        :int(len(downstream_emb) * threshold)]

    # print(puesdo_normal_emb)
    similarity_center =l2diff(torch.tensor(avg_after),torch.tensor(avg_before))
    total_normal_emb= np.concatenate((puesdo_normal_emb, pre_emb))
    r_distribution_before = r_distirbution_fit(torch.tensor(puesdo_normal_emb), torch.tensor(avg_before))

    r_distribution_after = r_distirbution_fit(torch.tensor(total_normal_emb), torch.tensor(avg_before))
    theta_distribution_before = theta_distribution_fit3(puesdo_normal_emb, avg_before)
    theta_distribution_after = theta_distribution_fit3(total_normal_emb, avg_before)
    r_theta_before = np.concatenate((r_distribution_before, theta_distribution_before), axis=1)
    r_theta_after = np.concatenate((r_distribution_after, theta_distribution_after), axis=1)
    similarity_r_theta = distribution_sim(torch.tensor(r_theta_before),torch.tensor(r_theta_after))
    uniformity = cal_uniformity(torch.tensor(total_normal_emb),torch.tensor(avg_before)).mean()
    return similarity_center.item(),similarity_r_theta.item(),uniformity.item()
