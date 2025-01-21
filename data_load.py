import os
import torch
import dgl
from dgl import DGLGraph
import torch_geometric
import networkx as nx
from data_util_1 import to_dgl
external_datasets_small_unlabled = ["cora","pubmed","wikics","reddit"]
external_datasets_large_unlabled = ["Electronics","Entertainment","Fashion","Home","Learning","arxiv"]
external_datasets_labled = ["instagram","yelpres","yelpnyc"]
down_datasets = ['c15','amazon_cn','Tolokers','twi20','Enron','yelphtl']
def load_external_data(data_name, external_path="/home/caoyuxuan/graphmae/text_data/pre/", ano='0', type=0):
    if type == 0:
        external_path = external_path + "unlabeled/small/" + data_name + "/"
    elif type == 1:
        external_path = external_path + "labeled/" + data_name + "/"
    elif type == 2:
        external_path = external_path + "unlabeled/big/" + data_name + "/raw/"
    elif type == 3:
        external_path = external_path + "self/" + data_name + "/"
    external_datas = os.listdir(external_path)
    external_datas = [data for data in external_datas ]
    graph_pres = []
    for external_data in external_datas:
        pyg_data = torch.load(external_path + external_data)
        print(pyg_data)
        if ano == '1':
            pyg_data.y = torch.tensor(pyg_data.y)
        graph_pres.append(pyg_data)
    return graph_pres, external_datas

def load_external_data_big(data_name,external_path = "graphmae/text_data/pre/",ano='0',type=0):
    external_path = external_path + "unlabeled/big/" + data_name + "/"
    external_datas = os.listdir(external_path + "raw/")
    graph_pres = []
    for external_data in external_datas:
        if "subgraph" in external_data:
            continue
        print(external_data)
        print(external_path + external_data)
        pyg_data = torch.load(external_path + "raw/" + external_data)
        if ano == '1':
            pyg_data.y = torch.tensor(pyg_data.y)
        if isinstance(pyg_data, DGLGraph):
            graph_pre = pyg_data
        elif isinstance(pyg_data, torch_geometric.data.Data):
            graph_pre = to_dgl(pyg_data)
        print(pyg_data)
        graph_pres.append(graph_pre)
    ego_graphs_file_path = external_path + "subgraphs/"
    return graph_pres, external_datas, ego_graphs_file_path
def load_raw_data(data_name, external_path="graphmae/text_data/pre/", ano='0', type=0):
    if type == 0:
        external_path = external_path + "unlabeled/small/" + data_name + "/"
    elif type == 1:
        external_path = external_path + "labeled/" + data_name + "/"
    elif type == 2:
        external_path = external_path + "unlabeled/big/" + data_name + "/raw/"
    elif type == 3:
        external_path = external_path + "self/" + data_name + "/"
    print(external_path)
    external_data = data_name + ".pt"
    pyg_data = torch.load(external_path + external_data)
    if ano == '1':
        pyg_data.y = torch.tensor(pyg_data.y)
    return pyg_data
def load_down_data(data_name,down_path="graphmae/text_data/down/"):
    down_path = down_path + data_name + ".pt"
    pyg_data = torch.load(down_path)
    pyg_data.x = torch.tensor(pyg_data.x)
    pyg_data.y = torch.tensor(pyg_data.y)
    print(pyg_data)
    return pyg_data
def statistic_cal(data):
    if isinstance(data, torch_geometric.data.Data):
        nxgraph = torch_geometric.utils.to_networkx(data)
    elif isinstance(data, nx.Graph):
        nxgraph = data
    elif isinstance(data,dgl.DGLGraph ):
        nxgraph = data.to_networkx()
    elif isinstance(data, DGLGraph):
        nxgraph = data.to_networkx()
    else:
        raise ValueError("data type error")
    node_num = len(list(nxgraph.nodes))
    edge_num = len(list(nxgraph.edges))
    return node_num,edge_num
# write a testing sample:
# external_path = "/home/caoyuxuan/graphmae/text_data/pre/"
# for data_name in external_datasets_small_unlabled:
#     graph_pres, external_datas = load_external_data(data_name, external_path)
#     for i in range(len(graph_pres)):
#         print(external_datas[i])
#         data = graph_pres[i]
#         print(statistic_cal(data))

external_path = "/home/caoyuxuan/graphmae/text_data/pre/"
for data_name in down_datasets:
    graph_pres, external_datas = load_external_data(data_name, external_path, type=3)
    for i in range(len(graph_pres)):
        print(external_datas[i])
        data = graph_pres[i]
        print(statistic_cal(data))

# external_path = "/home/caoyuxuan/graphmae/text_data/pre/"
# for data_name in external_datasets_large_unlabled:
#     graph_pres, external_datas = load_external_data(data_name, external_path, type=2)
#     for i in range(len(graph_pres)):
#         print(external_datas[i])
#         data = graph_pres[i]
#         print(statistic_cal(data))

# down_path = "/home/caoyuxuan/graphmae/text_data/down/"
# for data_name in down_datasets:
#     data = load_down_data(data_name, down_path)
#     print(statistic_cal(data))

# external_path = "/home/caoyuxuan/graphmae/text_data/pre/"
# for data_name in external_datasets_labled:
#     graph_pres, external_datas = load_external_data(data_name, external_path, type=1)
#     for i in range(len(graph_pres)):
#         print(external_datas[i])
#         data = graph_pres[i]
#         print(statistic_cal(data))
