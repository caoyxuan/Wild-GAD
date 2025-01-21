import os
import csv
import random
import warnings
import argparse
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pygod.detector import (DOMINANT, AnomalyDAE, AdONE, GAAN, CONAD, CoLA, DONE, GAE, OCGNN, Radar, SCAN, ANOMALOUS)
from pygod.metric import eval_roc_auc, eval_average_precision
from data_util.data_util_1 import scale_feats
from distance.distance import cal_distance_unlabeled, cal_distance_labeled
from data_util.data_load import load_down_data,load_external_data
warnings.filterwarnings('ignore')
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--down_data', type=str)
    parser.add_argument('--model', type=str, default='DOMINANT')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_path', type=str, default="outs/continue/perf")
    parser.add_argument('--base_model_path', type=str, default="outs/base/models/")
    parser.add_argument('--con_model_path', type=str, default="outs/continue/models/")
    parser.add_argument('--down_path', type=str, default="/text_data/down/")
    return parser.parse_args()

# Evaluation function
def my_eval(dataset_name, data, y_true, score):
    if dataset_name == "twi20":
        mask = (data.y == -1)
        y_true = y_true[~mask]
        score = score[~mask]
        mask_twice = torch.isnan(score)
        return (
            eval_roc_auc(y_true[~mask_twice], score[~mask_twice]),
            eval_average_precision(y_true[~mask_twice], score[~mask_twice])
        )
    return eval_roc_auc(y_true, score), eval_average_precision(y_true, score)

# Main processing
args = build_args()
pre_name = ["cora"]
down_name = ['amazon_cn']
batch_models = ["OCGNN"]

lrs = {"twi20": 0.01, "amazon_cn": 0.0001, "c15": 0.00001, "Enron": 0.01, "yelphtl": 0.0001, "Tolokers": 0.0001}
hid_dims = {"twi20": 256, "amazon_cn": 128, "c15": 128, "Enron": 256, "yelphtl": 256, "Tolokers": 256}
lrs_con = {"twi20": 0.01, "amazon_cn": 0.0001, "c15": 0.00001, "Enron": 0.001, "ye lphtl": 0.0001, "Tolokers": 0.0001}

pre_datasets_small_unlabeled = ["cora", "pubmed", "wikics", "reddit"]
pre_datasets_large_unlabeled = ["arxiv", "Electronics", "Entertainment", "Fashion", "Home", "Learning"]
lambda_ewc = [0.2,0.4,0.6,0.8]
AUC_dataframe = pd.DataFrame(columns=batch_models, index=down_name)
AP_dataframe = pd.DataFrame(columns=batch_models, index=down_name)
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for name in down_name:
    for detector_name in batch_models:
        data = load_down_data(name, down_path=args.down_path)
        y_true = torch.tensor(data.y)
        detector = OCGNN
        torch.manual_seed(0)
        for trial in tqdm(range(1)):
            if name == "amazon_cn":
                data.x = scale_feats(data.x)
                model = detector(num_neigh=8,lr=0.0001,epochs =100,gpu=args.gpu,save_emb=False,model_path = args.base_model_path + name + "/" + detector_name + "2.pt") #,hid_dim=random.choice([128])
            else:
                data.x = scale_feats(data.x)
                model = detector(lr=lrs[name], epochs=100, gpu=args.gpu, hid_dim=hid_dims[name], save_emb=False,
                             model_path=args.base_model_path + name + "/" + detector_name + ".pt")
            # model.fit(data)
            # model.compute_fisher(data)
            # if not os.path.exists(args.base_model_path + name):
            #     os.makedirs(args.base_model_path + name)
            # # print(args.base_model_path + name + "/" + detector_name + ".pt")
            # model.model.save_model(args.base_model_path + name + "/" + detector_name + "3.pt")
            model.init_gnn_model()
            model.model.load_model(args.base_model_path + name + "/" + detector_name + "3.pt")
            score = model.new_predict(data)
            in_auc, in_ap = my_eval(name, data, y_true, score)
            print("original auc:{} original ap:{}".format(in_auc, in_ap))
            emb1 = model.get_emb(data)
            for pre in pre_name:
                distances = []
                data_type = 0 if pre in pre_datasets_small_unlabeled else 2 if pre in pre_datasets_large_unlabeled else 3
                graph_pres, pre_datas = load_external_data(pre, type=data_type)
                for idx, pre_data in enumerate(graph_pres):
                    try:
                        pre_data.num_nodes = pre_data.x.shape[0] 
                        pre_data.x = scale_feats(pre_data.x)
                        if data_type==0:
                            emb2= model.get_emb(pre_data)
                        elif data_type == 2:
                            emb     2 = model.batch_get_emb(pre_data)
                        sim_center, sim_r_theta_normal, uni_normal = cal_distance_unlabeled(emb1.numpy(),
                                                                                                emb2.numpy())
                        distances.append([pre + pre_datas[idx], sim_center, sim_r_theta_normal, uni_normal])
                    except Exception as e:
                        print(e)
                with open(args.out_path + name + "_distancenew3.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(distances)
