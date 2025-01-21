import numpy as np
import torch
import math
import networkx as nx
import numpy as np
import torch_geometric
import torch_geometric.utils 
from torch_geometric.utils import index_to_mask
from data_load import load_external_data
def from_dgl(g):
    data = Data()
    data.edge_index = torch.stack(g.edges(), dim=0)
    print(data.is_undirected())
    for attr, value in g.ndata.items():
        data[attr] = value
    for attr, value in g.edata.items():
        data[attr] = value
    return data
def drop_nodes(data, aug_ratio, labels ):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    edge_mask = np.array(
        [n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.y = data.y[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data
    return data


def drop_edges(data, aug_ratio, labels ):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    labels = data.y
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(edge_num, edge_num - permute_num)
    edge_index = edge_index[:, idx_add]
    data.edge_index = torch.tensor(edge_index)
    return data


def permute_edges(data, aug_ratio, labels ):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))
    edge_index = np.concatenate(
        (edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data


def mask_nodes(data, aug_ratio, labels ):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data

def subgraph2(data, aug_ratio, labels ):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)
    edge_index = data.edge_index.numpy()
    subset = np.random.choice(node_num, sub_num, replace=False)
    # array to list:
    subset = subset.tolist()
    edge_index,edge_attr = torch_geometric.utils.subgraph(torch.tensor(subset, dtype=torch.long), data.edge_index, relabel_nodes=True)
    subset = torch.tensor(subset, dtype=torch.long)
    node_mask = index_to_mask(subset, node_num) if subset.dtype != torch.bool else subset
    try:
        data.edge_index = edge_index
        data.num_nodes = sub_num
        data.x = data.x[node_mask]
        data.y = data.y[node_mask] if hasattr(data, 'y') else None
        data.edge_attr = edge_attr
    except:
        print("error")
        data = data
    return data
def subgraph(data, aug_ratio, labels ):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array(
        [n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

    edge_index = data.edge_index.numpy()
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.y = data.y[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data
def process(ogb, aug, ratio):
    file_path_pre = "/home/caoyuxuan/graphmae/text_data/pre/labeled/"
    file_path_load = "/home/caoyuxuan/newanomaly/data/pre/"
    data = torch.load(file_path_pre +ogb+"/"+ ogb +".pt")
    if not isinstance(data, Data):
        data = from_dgl(data)
    dir_path = file_path_pre + ogb + "/"
    data.y = torch.tensor(data.y)
    print(dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    data_aug = None
    data1 = data.clone()
    if aug == "drop_edges":
        data_aug = drop_edges(data1, ratio, None)
    elif aug == "drop_nodes":
        data_aug = drop_nodes(data1, ratio, None)
    elif aug == "subgraph2":
        data_aug = subgraph2(data1, ratio, None)
    elif aug == "mask_nodes":
        data_aug = mask_nodes(data1, ratio, None)
    torch.save(data_aug, dir_path + aug + str(ratio) + "new.pt")
    print(data_aug)
    print(f'aug:{aug}, ratio:{ratio},')
def statistic_cal(data):
    nxgraph = torch_geometric.utils.to_networkx(data)
    density = nx.density(nxgraph)
    degrees = nx.degree_histogram(nxgraph)
    avg_degree = float(sum(degrees[i] * i for i in range(len(degrees))) / (nxgraph.number_of_nodes()))
    std = math.sqrt(
        sum(math.pow(i - avg_degree, 2) * degrees[i] for i in range(len(degrees))) / (nxgraph.number_of_nodes()))
    # closeness_centrality = sum(nx.closeness_centrality(nxgraph).values()) / len(nxgraph.node)
    # print(closeness_centrality)
    degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(nxgraph)
    avg_clu_co = nx.average_clustering(nxgraph)
    transitivity = nx.transitivity(nxgraph)
    topo_vec = [ avg_degree,std,density, degree_pearson_correlation_coefficient,
               transitivity, avg_clu_co]
    return topo_vec

# if __main__ == "__main__":
#     data = torch_geometric.datasets.PPI(root='/tmp/PPI', split='train')[0]
#     data = subgraph(data, 0.5, None)
#     print(statistic_cal(data))
