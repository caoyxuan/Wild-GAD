from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import json
from with_bert import get_vec
import torch
from graphmae.datasets.data_util import to_dgl
d_name = "ogbn-products"
dataset = PygNodePropPredDataset(name = d_name, root = "text_data/")
data = dataset[0]
print(data.num_nodes)
pre_process_str_mapping = []
Home_Lifestyle = [0, 5,9, 17,21, 23, 25, 26, 37, 38,42, 45]#12
Learning = [4,18,19, 22,36, 43, 44]#7
Personal_Fashion= [1, 2,10, 11, 28,29, 30, 33] #8
Entertainment= [3, 6, 7, 13, 15, 20, 27, 35] #8
Electronics = [8,12, 14, 31, 32, 34,39, 40, 41] #9
Miscellaneous = [  16, 24,  46]
with open("matches.json", "r") as json_file:
    matches = json.load(json_file)

def get_node_indices(data, type):
    indices = [np.where(data.y == i)[0] for i in type]
    combined_result = []
    for sublist in  indices:
        combined_result.extend(sublist)
    return combined_result
Personal_Fashion_node_indices = get_node_indices(data, Personal_Fashion)
Learning_node_indices = get_node_indices(data, Learning)
Home_Lifestyle_node_indices = get_node_indices(data, Home_Lifestyle)
Entertainment_node_indices = get_node_indices(data, Entertainment)
Electronics_node_indices = get_node_indices(data, Electronics)
Miscellaneous_node_indices = get_node_indices(data, Miscellaneous)
type_group_indices = {
    "Home_Lifestyle": Home_Lifestyle_node_indices,
    "Learning": Learning_node_indices,
    "Personal_Fashion": Personal_Fashion_node_indices,
    "Entertainment": Entertainment_node_indices,
    "Electronics": Electronics_node_indices,
    "Miscellaneous": Miscellaneous_node_indices
}
type_groups = ["Home_Lifestyle", "Learning", "Personal_Fashion", "Entertainment", "Electronics", "Miscellaneous"]

import numpy as np
def get_match_type(node_index, matches):
    if str(node_index) in matches['trn_matches']:
        return 'trn_matches'
    elif str(node_index) in matches['tst_matches']:
        return 'tst_matches'
    else:
        return 'Node index not found in either trn_matches or tst_matches'
from multiprocessing import Pool

from tqdm import tqdm
matches_dict = {}
for match_type in ['trn_matches', 'tst_matches']:
    for match in tqdm(matches[match_type]):
        matches_dict[match['node_idx']] = {'title': match['title'], 'content': match['content'], 'type': match_type}
def process_title(node_index):
    match_data = matches_dict.get(str(node_index))
    if match_data is not None:
        return f"{match_data['title']} "
        # return { 'title': match_data['title'], 'content': match_data['content'] }
    else:
        return None

def process_node(node_index):
    match_data = matches_dict.get(str(node_index))
    if match_data is not None:
        return f"{match_data['title']} {match_data['content']}"
        # return { 'title': match_data['title'], 'content': match_data['content'] }
    else:
        return None
with Pool() as p:
    raw_titles = p.map(process_title, range(data.num_nodes))

with Pool() as p:
    raw_texts = p.map(process_node, range(data.num_nodes))
