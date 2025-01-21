
from collections import namedtuple, Counter
import numpy as np
import dgl
from torch_geometric.data import Data, HeteroData
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import load_graphs
from sklearn.preprocessing import StandardScaler
import os
import random

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}
ano_dataset_dgl = ['tolokers','questions']
supervised_ano_dataset =  ['tolokers','questions']
unsupervised_ano_dataset = ['books', 'disney','enron','reddit','weibo','inj_cora','inj_amazon']
cross_dataset = ['Amazon','YelpHotel','YelpNYC','YelpRes']
def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)
def to_dgl(
    data: Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']
) -> Any:
    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, store in data.edge_items():
            if store.get('edge_index') is not None:
                row, col = store.edge_index
            else:
                row, col, _ = store['adj_t'].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, store in data.node_items():
            for attr, value in store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
from os.path import join
import scipy.io as sio
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import to_undirected

def load_mat(file_dir, fn):
    fp = join(file_dir, fn)
    data = sio.loadmat(fp)
    return {
        "features": sp.lil_matrix(data['Attributes']),
        "adj": sp.csr_matrix(data['Network']),
        "ad_labels": np.squeeze(np.array(data['Label']))
    }


def mat_to_pyg_data(data, undirected=False):
    features = torch.from_numpy(data["features"].todense()).float()

    adj = data["adj"]
    edge_index, _ = from_scipy_sparse_matrix(adj)

    ad_labels = data['ad_labels']

    if undirected:
        print("Processing the graph as undirected...")
        if data.is_directed():
            edge_index = to_undirected(data.edge_index)

    data = Data(x=features, edge_index=edge_index)
    if undirected:
        assert data.is_undirected()
    return data, ad_labels
def cal_graph_statistics1(g):
    import dgl.function as fn
    in_deg = g.in_degrees().view(-1, 1).float()
    out_deg = g.out_degrees().view(-1, 1).float()
    g.ndata['in_deg'] = in_deg
    g.ndata['out_deg'] = out_deg
    # g.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'neigh'))
    # in_similarity = F.cosine_similarity(g.ndata['feat'], g.ndata['neigh'], dim=1)
    # rev_g = dgl.reverse(g, copy_ndata='True')
    # rev_g.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'neigh'))
    # out_similarity = F.cosine_similarity(rev_g.ndata['feat'], rev_g.ndata['neigh'], dim=1)
    # g.ndata['in_sim'] = in_similarity
    # print(g.ndata['in_sim'])
    # g.ndata['out_sim'] = out_similarity
    # print( g.ndata['out_sim'])
    out_homophily_values,in_homophily_values = [],[]
    in_feat_values,out_feat_values = [], []
    deg_2_in,deg_2_out = [],[]
    for i in range(g.number_of_nodes()):
        successors = g.successors(i).tolist()
        # two_hop_successors = set(g.successors(successors).flatten().tolist())
        # two_hop_predecessors = set(g.successors(successors).flatten().tolist())
        two_hop_successors = len(set(sum([g.successors(successors[k]).flatten().tolist() for k in range(len(successors))], [])))
        deg_2_in.append(two_hop_successors)
        predecessors = g.predecessors(i).tolist()# 获取邻居节点列表
        two_hop_predecessors = len(
            set(sum([g.successors(predecessors[k]).flatten().tolist() for k in range(len(predecessors))], [])))
        deg_2_out.append(two_hop_predecessors)
        mean_feat_successor = F.cosine_similarity(g.ndata['feat'][successors],g.ndata['feat'][i].unsqueeze(0),dim=1,eps=1e-5).mean() if len(successors) > 0 else -1
        mean_feat_predecessors = F.cosine_similarity(g.ndata['feat'][predecessors],g.ndata['feat'][i].unsqueeze(0),dim=1,eps=1e-5).mean() if len(predecessors) > 0 else -1
        out_feat_values.append(mean_feat_predecessors)
        in_feat_values.append(mean_feat_successor)
        similar_successors  = sum(g.ndata['label'][successors] == g.ndata['label'][i])  # 计算相似标签的邻居数量
        similar_predecessors  = sum(g.ndata['label'][ predecessors] == g.ndata['label'][i])  # 计算相似标签的邻居数量
        out_homophily = similar_successors / len(successors) if len(successors) > 0 else -1  # 计算 homophily 值
        out_homophily_values.append(out_homophily)
        in_homophily =  similar_predecessors/ len(predecessors) if len(predecessors) > 0 else -1
        in_homophily_values.append(in_homophily)
    g.ndata['in_homophily'] =  torch.tensor(in_homophily_values)
    g.ndata['out_homophily'] =torch.tensor( out_homophily_values)
    g.ndata['in_sim1'] = torch.tensor(in_feat_values)
    g.ndata['deg_2_in'] = torch.tensor(deg_2_in)
    g.ndata['deg_2_out'] = torch.tensor(deg_2_out)
    print(g.ndata['in_sim1'])
    g.ndata['out_sim1'] = torch.tensor(out_feat_values)
    print( g.ndata['out_sim1'])
def cal_graph_statistics(g):
    import dgl.function as fn
    # g = dgl.to_bidirected()
    deg = g.in_degrees().view(-1, 1).float()
    rw_pe= dgl.random_walk_pe(g,16)
    lp_pe = dgl.laplacian_pe(g,16)
    g.ndata['deg'] = deg
    g.ndata['rw_pe'] = rw_pe
    g.ndata['lp_pe'] = lp_pe
    g.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'neigh'))
    node_features = g.ndata['feat']
    neighbor_features = g.ndata['neigh']
    # g.update_all(fn.copy_u('rw_pe', 'm'), fn.mean('m', 'neigh_rw'))
    # neighbor_rw_features = g.ndata['neigh_rw']
    # g.update_all(fn.copy_u('lp_pe', 'm'), fn.mean('m', 'neigh_lp'))
    # neighbor_lp_features = g.ndata['neigh_lp']
    similarity = F.cosine_similarity(node_features, neighbor_features, dim=1)
    distance = torch.sqrt(torch.sum(torch.pow( node_features - neighbor_features, 2), 1))
    # rw_pe_distance =torch.sqrt(torch.sum(torch.pow(  rw_pe - neighbor_rw_features, 2), 1))
    # lp_pe_distance = torch.sqrt(torch.sum(torch.pow(lp_pe - neighbor_lp_features, 2), 1))
    g.ndata['neigh_sim_feat'] = similarity
    # g.ndata['rw_pe_distance'] = rw_pe_distance
    # g.ndata['lp_pe_distance'] = lp_pe_distance
    g.ndata['neigh_distance'] = distance
    clustering_coefficients = []
    homophily_values = []
    # import networkx as nx
    # nx_g = g.to_networkx()
    # print(nx_g)
    # # Calculate clustering coefficients
    # clustering_coefficients = nx.clustering(nx_g)
    #
    # # Convert to PyTorch tensor
    # clustering_coefficients_tensor = torch.tensor(list(clustering_coefficients.values()))
    # print(clustering_coefficients_tensor)
    for i in range(g.number_of_nodes()):
        neighbors = g.successors(i).tolist()  # 获取邻居节点列表
        similar_neighbors = sum(g.ndata['label'][neighbors] == g.ndata['label'][i])  # 计算相似标签的邻居数量
        homophily = similar_neighbors / len(neighbors) if len(neighbors) > 0 else 0  # 计算 homophily 值
        homophily_values.append(homophily)
        # num_neighbors = len(neighbors)

        # if num_neighbors < 2:
        #     clustering_coefficients.append(0.0)
        # else:
        #     num_triangles = 0
        #     for i in range(num_neighbors - 1):
        #         for j in range(i + 1, num_neighbors):
        #             # Check if there is an edge between the neighbors[i] and neighbors[j]
        #             if neighbors[i] in g.predecessors(neighbors[j]):
        #                 num_triangles += 1
            # clustering_coefficients.append(2.0 * num_triangles / (num_neighbors * (num_neighbors - 1)))
    # print(clustering_coefficients)
    # g.ndata['clustering'] = torch.tensor(clustering_coefficients)
    g.ndata['homophily'] = torch.tensor(homophily_values)
    return g
def load_transfer_dataset_pretrain(dataset_name,normal_only = False,train_only = False,semi=True,infer = False,abnormal_mask=False,balance_sample=False,normal_rate = 1,abnormal_rate =1,sync_style = None,sync_num=None,sync_seed = None):
    if dataset_name in unsupervised_ano_dataset:
        file_path = "/home/caoyuxuan/new_GCC/data/" + dataset_name + ".pt"
        data = torch.load(file_path)
        num_nodes = data.num_nodes
        g = to_dgl(data)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop( g )
        g  = dgl.to_simple( g )
        g.ndata["label"] =g.ndata['y']
        print(g.ndata['label'])
        g.ndata['feat'] = g.ndata['x']
        # if attr_plain:
        #     g.ndata['feat'] = torch.ones_like(data.x)
        g.ndata['feat'] = scale_feats(g.ndata['x'])
        labels = g.ndata['y']
        num_classes = int(max(labels)) + 1
        num_features = g.ndata['x'].shape[1]
        # if normal_only == True:
        #     leg_idx = np.squeeze(np.nonzero(labels == 0))
        #     g = dgl.node_subgraph(g, leg_idx)
        if balance_sample == True:
            abnormal_idx = np.squeeze(np.nonzero(labels == 1))
            normal_idx = np.squeeze(np.nonzero(labels == 0))
            sampled_len = len(abnormal_idx)
            perm_normal = torch.randperm(len(normal_idx))
            perm_abnormal = torch.randperm(len(abnormal_idx))
            sampled_normal_nodes = normal_idx[perm_normal[:int(len(normal_idx) * normal_rate)]]
            sampled_abnormal_nodes = abnormal_idx[perm_abnormal[:int(len(abnormal_idx) * abnormal_rate)]]
            total_sampled_idxs = sampled_normal_nodes + sampled_abnormal_nodes
            # sampled_nodes = perm[: sampled_len]
            # normal_sampled_idxs = normal_idx[sampled_nodes]
            # total_sampled_idxs = normal_sampled_idxs + abnormal_idx
            return_g = dgl.node_subgraph(g, total_sampled_idxs)
            return [return_g],num_features,num_classes
        for attr, value in g.ndata.items():
            print(attr)
        train_ratio = 0.3
        if train_only:
            if os.path.exists(dataset_name+"train_idxs.npy"):
                idx_train = np.load(dataset_name+"train_idxs.npy")
            else:

                index = list(range(g.num_nodes()))
                leg_idxs = np.squeeze(np.nonzero(labels == 0))

                idx_train, idx_rest, y_train, y_rest = train_test_split(leg_idxs.tolist(), labels[leg_idxs], stratify=labels[index],
                                                                            train_size=train_ratio,
                                                                        random_state=2, shuffle=True)
                np.save(dataset_name+"normalonlytrain_idxs.npy",idx_train)
                np.save(dataset_name + "normalonlytest_idxs.npy", idx_rest)
                if normal_only:
                    print(idx_train)
                    idx_train =idx_train[np.squeeze(np.nonzero(labels[idx_train] == 0))]
                    print(idx_train)
            g = dgl.node_subgraph(g, idx_train)
        # cal_graph_statistics1(g)

        return [g],num_features,num_classes
    elif "fix"  in dataset_name:
        graph = torch.load("/home/caoyuxuan/graphmae/" + dataset_name + ".pt")
        num_classes = int(max(graph.ndata["label"])) + 1
        num_features = graph.ndata["feat"].shape[1]
        return [graph],num_features,num_classes
    elif "sub" in dataset_name or "+" in dataset_name or "0." in dataset_name :
        data_names = dataset_name.split('+')
        # print(data_names)
        return_g=[]
        for data_name in data_names:
            if "0." in data_name:
                data = torch.load("/home/caoyuxuan/graphmae/cross_data/cross_dataset/aug_data/" + data_name + ".pt")
            elif "sub" in data_name :
                data = torch.load("/home/caoyuxuan/graphmae/cross_data/cross_dataset/cross_data/" + data_name + ".pt")
            else:
                matdata = load_mat("/home/caoyuxuan/graphmae/cross_data/cross_dataset/", data_name + ".mat")
                data, labels = mat_to_pyg_data(matdata)
                data.y = torch.tensor(labels)
            g = to_dgl(data)
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            g = dgl.to_simple(g)
            g.ndata["label"] = data.y
            g.ndata['feat'] = data.x
            g.ndata['feat'] = scale_feats(data.x)
            # cal_graph_statistics1(g)
            labels = data.y
            num_classes = int(max(data.y)) + 1
            num_features = data.x.shape[1]
            # print()
            # # print(g.num_nodes())
            # for attr, value in g.ndata.items():
            #     print(attr)
            if normal_only==True:
                leg_idx = np.squeeze(np.nonzero(labels == 0))
                g = dgl.node_subgraph(g, leg_idx)
            return_g.append(g)
            if abnormal_mask==True:
                leg_idx = np.squeeze(np.nonzero(labels == 1))
                g.ndata['feat'][leg_idx] = 0
            if balance_sample == True:
                abnormal_idx  =  np.squeeze(np.nonzero(labels == 1))
                normal_idx = np.squeeze(np.nonzero(labels == 0))
                sampled_len = len(abnormal_idx)
                perm_normal = torch.randperm(len(normal_idx))
                perm_abnormal = torch.randperm(len(abnormal_idx))
                sampled_normal_nodes = normal_idx[perm_normal[:int(len(normal_idx)*normal_rate)]]
                sampled_abnormal_nodes = abnormal_idx[perm_abnormal[:int(len(abnormal_idx) * abnormal_rate)]]
                total_sampled_idxs = sampled_normal_nodes+sampled_abnormal_nodes
                # sampled_nodes = perm[: sampled_len]
                # normal_sampled_idxs = normal_idx[sampled_nodes]
                # total_sampled_idxs = normal_sampled_idxs + abnormal_idx
                return_g = dgl.node_subgraph(g,total_sampled_idxs)
                # if infer==False:
        #     cal_graph_statistics1(g)
        return return_g, num_features, num_classes
    elif dataset_name in cross_dataset:
        matdata = load_mat("/home/caoyuxuan/graphmae/cross_data/cross_dataset/", dataset_name+".mat")
        data,labels = mat_to_pyg_data(matdata)
        data.y = torch.tensor(labels)
        # data = torch.load("/home/caoyuxuan/graphmae/cross_data/cross_dataset/"+dataset_name+"sub_0.pt")
        g = to_dgl(data)
        g = dgl.remove_self_loop(g)
        g =dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        g.ndata["label"] = data.y
        g.ndata['feat'] = data.x
        #
        g.ndata['feat'] = scale_feats(data.x)
        num_classes = int(max(labels)) + 1
        num_features =data.x.shape[1]
        normals = np.squeeze(np.nonzero(labels == 0))
        abnormals = np.squeeze(np.nonzero(labels == 1))

        if abnormal_mask == True:
                g.ndata['feat'][abnormals] = 0
        # if up_sample == True:
        if balance_sample == True:
            abnormal_idx = np.squeeze(np.nonzero(labels == 1))
            normal_idx = np.squeeze(np.nonzero(labels == 0))
            sampled_len = len(abnormal_idx)
            perm_normal = torch.randperm(len(normal_idx))
            perm_abnormal = torch.randperm(len(abnormal_idx))
            sampled_normal_nodes = normal_idx[perm_normal[:int(len(normal_idx) * normal_rate)]]
            sampled_abnormal_nodes = abnormal_idx[perm_abnormal[:int(len(abnormal_idx) * abnormal_rate)]]
            total_sampled_idxs = sampled_normal_nodes.tolist() + sampled_abnormal_nodes.tolist()
            print( len(total_sampled_idxs))
            # sampled_nodes = perm[: sampled_len]
            # normal_sampled_idxs = normal_idx[sampled_nodes]
            # total_sampled_idxs = normal_sampled_idxs + abnormal_idx
            return_g = dgl.node_subgraph(g, total_sampled_idxs)
            return [return_g],  num_features, num_classes
        if normal_only==True:
            leg_idx = np.squeeze(np.nonzero(labels == 0))
            g = dgl.node_subgraph(g, leg_idx)
        # if infer == False:
        #     cal_graph_statistics1(g)
        for attr, value in g.ndata.items():
            print(attr)
        return [g],  num_features, num_classes
    elif dataset_name in supervised_ano_dataset:
        file_path = "/home/caoyuxuan/new_GCC/datasets/" + dataset_name
        graph = load_graphs(file_path)[0][0]
        # graph = dgl.add_self_loop(graph)
        graph = dgl.to_simple(graph)
        for attr, value in graph.ndata.items():
            print(attr)
        graph.ndata['feat'] = graph.ndata['feature'].float()
        graph.ndata['label'] = graph.ndata['label'].long()
        labels = graph.ndata["label"]
        print(graph.ndata["feature"])
        # cal_graph_statistics1(graph)
        return [graph]

    elif "dgl" in dataset_name:
        return_g = []
        pre_text_data_path = "/home/caoyuxuan/graphmae/text_data/"
        for data_name in dataset_name.split('+'):
            g = torch.load(pre_text_data_path + data_name + ".pt")
            print(g.ndata['feat'])
        g.ndata['feat'] = scale_feats(g.ndata['feat'])
        g.ndata['feat'] =torch.cat((g.ndata['feat'], g.ndata['rw_feat']), axis=1)

        num_feat = len(g.ndata['feat'][0])
        num_classes = 2
        return_g.append(g)
        return return_g, num_feat,num_classes
    elif "synthetic" in dataset_name:
        g = torch.load("/home/caoyuxuan/graphmae/synthetic_data/new_synthetic/"+sync_style+"/"+sync_num+"_"+sync_seed+".pt")
        g.ndata['feat'] =g.ndata['pe']
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        return [g], 64, 2
    else:
        file_path = "/home/caoyuxuan/new_GCC/datasets/" + dataset_name
        graph = load_graphs(file_path)[0][0]
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = dgl.to_simple(graph)
        for attr, value in graph.ndata.items():
            print(attr)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['label'] = graph.ndata['label'].long()
        labels = graph.ndata["label"]
        print(graph.ndata["feature"])
        graph.ndata["feat"] = scale_feats(graph.ndata["feature"])
        print(  graph.ndata["feat"])
        num_classes = int(max( labels)) + 1
        num_features = graph.ndata['feature'].shape[1]
        # train_ratio, val_ratio = 0.4, 0.2
        leg_idx = np.squeeze(np.nonzero(labels == 0))

        if dataset_name == "tsocial":
            normal_only = True
            train_only = True
            n= graph.num_nodes()
            index = list(range(graph.num_nodes()))
            train_mask = torch.zeros([len(labels)]).bool()
            val_mask = torch.zeros([len(labels)]).bool()
            test_mask = torch.zeros([len(labels)]).bool()
            train_ratio = 0.05
            val_ratio = 0.2
            samples = 20
            # for i in range(1):
            seed = 3407
            set_seed(seed)
            idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                    train_size=train_ratio, random_state=seed,
                                                                    shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    train_size=int(len(index) * val_ratio),
                                                                    random_state=seed, shuffle=True)
            train_mask[idx_train] = 1
            val_mask[idx_valid] = 1
            test_mask[idx_test] = 1
            print("process pre data done1")
            graph.ndata['train_masks'] = train_mask
            graph.ndata['val_masks'] = val_mask
            graph.ndata['test_masks'] = test_mask
            torch.save(graph,"tosical_{}.pt".format(train_ratio))
        else:
            if semi== True:
                train_mask = torch.zeros([len(labels)]).bool()
                val_mask = torch.zeros([len(labels)]).bool()
                test_mask = torch.zeros([len(labels)]).bool()
                train_ratio = 0.01
                index = list(range(graph.num_nodes()))
                idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                        train_size=train_ratio,
                                                                        random_state=2, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                        test_size=0.67,
                                                                        random_state=2, shuffle=True)
                train_mask[idx_train] = 1
                val_mask[idx_valid] = 1
                test_mask[idx_test] = 1
                graph.ndata['train_masks'] = train_mask
                graph.ndata['val_masks'] = val_mask
                graph.ndata['test_masks'] = test_mask
            graph.ndata['val_mask'] = graph.ndata['val_masks']
            graph.ndata['train_mask'] = graph.ndata['train_masks']
            graph.ndata['test_mask'] = graph.ndata['test_masks']
            train_mask = graph.ndata['train_masks']
    if normal_only == True:
        if train_only == True:
            train_nid = np.unique(np.nonzero(train_mask.data.numpy())[0].astype(np.int64))
            print(len(train_nid))
            leg_train_nid = np.squeeze(np.nonzero(labels[train_nid] == 0))
            new_graph = dgl.node_subgraph(graph, leg_train_nid)
            train_dataloader = [new_graph]
            eval_train_dataloader = [new_graph]
            print("leg train only!")
        # pyg_data = from_dgl(graph )
        # torch.save(pyg_data, dataset_name + "processed.pt")
        else:
            new_graph = dgl.node_subgraph(graph, leg_idx)
            train_dataloader = [new_graph]
            eval_train_dataloader = [new_graph]
            # train_dataloader = GraphDataLoader([graph], batch_size=64)
            # eval_train_dataloader =train_dataloader
            # valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            # test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("process pre data done2")
    else:
        if train_only == True:
            train_nid = np.unique(np.nonzero(train_mask.data.numpy())[0].astype(np.int64))
            print(len(train_nid))
            new_graph = dgl.node_subgraph(graph, train_nid)
            train_dataloader = [new_graph]
            eval_train_dataloader = [new_graph]
            print("train only!")
        else:
            train_dataloader = [graph]
            eval_train_dataloader =[graph]
    return train_dataloader, eval_train_dataloader, num_features, num_classes

def load_transfer_dataset_down(dataset_name,test_only = False,semi=False,attr_plain = False,normal_only=False,normal_rate = 1, abnoram_rate = 1,balance_sample=False,abnormal_rate =1,stru_attri = False):
    if dataset_name in unsupervised_ano_dataset:
        file_path = "/home/caoyuxuan/new_GCC/data/" + dataset_name + ".pt"

        data = torch.load(file_path)
        num_nodes = data.num_nodes
        g = to_dgl(data)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        g.ndata["label"] = g.ndata['y']
        if dataset_name == "inj_cora" or dataset_name == "inj_amazon":
            g.ndata["label"] =g.ndata['y'] >> 0&1
        g.ndata['feat'] = g.ndata['x']
        g.ndata['feat'] = data.x
        if stru_attri == True:
            rw_pe = dgl.random_walk_pe(g, 32)
            lp_pe = dgl.laplacian_pe(g, 32)
            pe = np.concatenate([rw_pe, lp_pe], axis=1)
            g.ndata['feat'] = torch.tensor(pe, dtype=torch.float32)
        else:
            g.ndata['feat'] = scale_feats(g.ndata['x'])
        labels = g.ndata['y']
        num_classes = int(max(labels)) + 1
        num_features = g.ndata['x'].shape[1]
        # print(g.ndata['label'])

        if normal_only == True:
            leg_idx = np.squeeze(np.nonzero(labels == 0))
            normal_g = dgl.node_subgraph(g, leg_idx)
            return [normal_g], [g]
        if test_only:
            test_idxs = np.load(dataset_name+"test_idxs.npy")
            g = dgl.node_subgraph(g,test_idxs)
        return [g],[g]
    elif "dgl" in dataset_name:
        return_g = []
        pre_text_data_path = "/home/caoyuxuan/graphmae/text_data/"
        for data_name in dataset_name.split('+'):
            g = torch.load(pre_text_data_path + data_name + ".pt")
            print(g.ndata['feat'])

        g.ndata['feat'] = scale_feats(g.ndata['feat'])
        g.ndata['feat'] =torch.cat((g.ndata['feat'], g.ndata['rw_feat']), axis=1)
        num_feat = len(g.ndata['feat'][0])
        num_classes = 2
        return_g.append(g)
        return return_g, return_g
    elif "fix" in dataset_name:
        graph = torch.load("/home/caoyuxuan/graphmae/" + dataset_name + ".pt")
        return [graph], [graph]
    elif "sub" in dataset_name :
        if "0." in dataset_name :
            data = torch.load("/home/caoyuxuan/graphmae/cross_data/cross_dataset/aug_data/" + dataset_name + ".pt")
        else:
            file_path = "/home/caoyuxuan/graphmae/cross_data/cross_dataset/cross_data/" + dataset_name + ".pt"
            data = torch.load(file_path)
        g = to_dgl(data)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        g.ndata["label"] = torch.tensor(data.y)
        # if attr_plain:
        #     g.ndata['feat'] = torch.ones_like(data.x)
        # else:
        # print(data.x.sum(1))
        g.ndata['feat'] = scale_feats(data.x)
        labels = data.y
        # print()
        num_classes = int(max(data.y)) + 1
        num_features = data.x.shape[1]
        # for attr, value in g.ndata.items():
        #     # print(attr)
        # if balance_sample == True:
        #     abnormal_idx = np.squeeze(np.nonzero(labels == 1))
        #     normal_idx = np.squeeze(np.nonzero(labels == 0))
        #     sampled_len = len(abnormal_idx)
        #     perm_normal = torch.randperm(len(normal_idx))
        #     perm_abnormal = torch.randperm(len(abnormal_idx))
        #     sampled_normal_nodes = normal_idx[perm_normal[:int(len(normal_idx) * normal_rate)]]
        #     sampled_abnormal_nodes = abnormal_idx[perm_abnormal[:int(len(abnormal_idx) * abnormal_rate)]]
        #     total_sampled_idxs = sampled_normal_nodes + sampled_abnormal_nodes
        #     # sampled_nodes = perm[: sampled_len]
        #     # normal_sampled_idxs = normal_idx[sampled_nodes]
        #     # total_sampled_idxs = normal_sampled_idxs + abnormal_idx
        #     return_g = dgl.node_subgraph(g, total_sampled_idxs)
        #     return [return_g], [g]
        #
        # if normal_only==True:
        #     leg_idx = np.squeeze(np.nonzero(labels == 0))
        #     print(leg_idx)
        #     print(np.squeeze(np.nonzero(labels == 1)))
        #     return_g = dgl.node_subgraph(g, leg_idx)

        return [g], [g]

    elif dataset_name in cross_dataset:
        matdata = load_mat("/home/caoyuxuan/graphmae/cross_data/cross_dataset/", dataset_name + ".mat")
        data, labels = mat_to_pyg_data(matdata)
        # print("here")
        g = to_dgl(data)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        g.ndata["label"] = torch.tensor(labels)
        g.ndata['feat'] = data.x
        if stru_attri == True:
            rw_pe = dgl.random_walk_pe(g, 32)
            lp_pe = dgl.laplacian_pe(g, 32)
            pe = np.concatenate([rw_pe, lp_pe], axis=1)
            g.ndata['feat'] = torch.tensor(pe, dtype=torch.float32)
        else:
            g.ndata['feat'] = scale_feats(data.x)
        # print(data.x.sum(1))
        # if balance_sample == True:
        #     abnormal_idx = np.squeeze(np.nonzero(labels == 1))
        #     normal_idx = np.squeeze(np.nonzero(labels == 0))
        #     sampled_len = len(abnormal_idx)
        #     perm_normal = torch.randperm(len(normal_idx))
        #     perm_abnormal = torch.randperm(len(abnormal_idx))
        #     sampled_normal_nodes = normal_idx[perm_normal[:int(len(normal_idx) * normal_rate)]]
        #     sampled_abnormal_nodes = abnormal_idx[perm_abnormal[:int(len(abnormal_idx) * abnormal_rate)]]
        #     total_sampled_idxs = sampled_normal_nodes + sampled_abnormal_nodes
        #     # sampled_nodes = perm[: sampled_len]
        #     # normal_sampled_idxs = normal_idx[sampled_nodes]
        #     # total_sampled_idxs = normal_sampled_idxs + abnormal_idx
        #     return_g = dgl.node_subgraph(g, total_sampled_idxs)
        #     return [return_g],[g]

        if normal_only == True:
            leg_idx = np.squeeze(np.nonzero(labels == 0))
            normal_g = dgl.node_subgraph(g, leg_idx)
            return [normal_g], [g]
        num_classes = int(max(labels)) + 1
        num_features = data.x.shape[1]
        # for attr, value in g.ndata.items():
        #     print(attr)
        return [ g], [g]
    elif dataset_name in supervised_ano_dataset:

        file_path = "/home/caoyuxuan/new_GCC/datasets/" + dataset_name
        graph = load_graphs(file_path)[0][0]
        labels = graph.ndata["label"]

        graph.ndata["feat"] = graph.ndata["feature"]

        # graph.ndata["feat"] = scale_feats(graph.ndata["feature"])
        if semi == True:
            train_mask = torch.zeros([len(labels)]).bool()
            val_mask = torch.zeros([len(labels)]).bool()
            test_mask = torch.zeros([len(labels)]).bool()
            train_ratio = 0.01
            index = list(range(graph.num_nodes()))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                    train_size=train_ratio,
                                                                    random_state=2, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    test_size=0.67,
                                                                    random_state=2, shuffle=True)
            train_mask[idx_train] = 1
            val_mask[idx_valid] = 1
            test_mask[idx_test] = 1
            graph.ndata['train_masks'] = train_mask
            graph.ndata['val_masks'] = val_mask
            graph.ndata['test_masks'] = test_mask
            graph.ndata['val_mask'] = graph.ndata['val_masks']
            graph.ndata['train_mask'] = graph.ndata['train_masks']
            graph.ndata['test_mask'] = graph.ndata['test_masks']

        else:
            graph.ndata['val_mask'] = graph.ndata['val_masks'][:, 0]
            graph.ndata['train_mask'] = graph.ndata['train_masks'][:, 0]
            graph.ndata['test_mask'] = graph.ndata['test_masks'][:, 0]


        if test_only:
            test_mask = graph.ndata['test_masks'][:,0]
            test_nid = np.nonzero(test_mask.data.numpy())[0].astype(np.int64)
            test_graph =  dgl.node_subgraph(graph, test_nid )
            test_dataloader = [test_graph]
            valid_dataloader = [test_graph]
            return valid_dataloader, test_dataloader,
        else:
            test_dataloader = [graph]
            valid_dataloader = [graph]
            return valid_dataloader, test_dataloader,

def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()
        print("new loaded")
        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
