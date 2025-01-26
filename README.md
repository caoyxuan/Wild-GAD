# Wild-GAD
Implementation of "*How to use Graph Data in the Wild to Help Graph Anomaly Detection?*"
## Brief Introduction

To effectively model the distribution of normal patterns and identify anomalies as deviations from these patterns, it is essential to provide the unsupervised model with a more comprehensive
view of normal behaviors. It enables the model to capture normal distributions more accurately, reducing the risk of misclassifying legitimate but underrepresented behaviors as anomalies. We propose borrowing knowledge from external data (i.e., graph data in the wild) to address such an issue. This leads to a natural question: **How can we use external data to help graph anomaly detection tasks?**

To answer this question, we design a framework called Wild-GAD. It primarily consists of three components:

1. **Diverse Database**: Ensures the model has access to a wide range of representative data resources.
2. **Data Selection Strategy**: Identifies the most useful external graph data for the task.
3. **Training Strategy**: Utilizes the selected external data to effectively perform the anomaly detection task.

Together, these components work cohesively to enhance graph anomaly detection by wisely leveraging external knowledge.

## Installation
We used the following packages under Python 3.10.
```
pytorch 1.4.0
torch-cluster 1.5.2
torch-geometric 1.0.3
torch-scatter 2.0.3
torch-sparse 0.5.1
torch-spline-conv 1.2.0
rdkit 2022.3.4
tqdm 4.31.1
tensorboardX 1.6
```
## Dataset Set-up

You can download all the external datasets  and downstream datasets from [here](https://modelscope.cn/datasets/WildGAD/WildGAD). It should be noted that you can also use our code to prepare your dataset. 

## Command-Line Execution
You can run the Wild-GAD framework using the following command:
```
python train.py [-h] [--down_data DOWN_DATA] [--model MODEL] [--gpu GPU] 
            [--out_path OUT_PATH] [--base_model_path BASE_MODEL_PATH] 
            [--con_model_path CON_MODEL_PATH] [--down_path DOWN_PATH] 
            [--lrs LRS] [--hid_dims HID_DIMS] [--lrs_con LRS_CON] 
            [--ewc_rate EWC_RATE]
```
#### Arguments:

- `--down_data DOWN_DATA`: Name of the downstream dataset.
- `--model MODEL`: Anomaly detection model to use.
- `--gpu GPU`: GPU ID for computation.
- `--out_path OUT_PATH`: Path to save the performance results.
- `--base_model_path BASE_MODEL_PATH`: Path to save the base model.
- `--con_model_path CON_MODEL_PATH`: Path to save the continued model.
- `--down_path DOWN_PATH`: Directory containing downstream datasets.
- `--lrs LRS`: Learning rates for datasets in a dictionary format`.
- `--hid_dims HID_DIMS`: Hidden dimensions for datasets in a dictionary format.
- `--lrs_con LRS_CON`: Learning rates for continued training in a dictionary format.
- `--ewc_rate EWC_RATE`: Elastic Weight Consolidation (EWC) rates in a dictionary format.

## Citation
If you use this code or dataset in your research, please cite our work:
```
@inproceedings{Cao2025WildGAD,
  title={How to use Graph Data in the Wild to Help Graph Anomaly Detection?},
  author={Yuxuan Cao and Jiarong Xu and Chen Zhao and Carl Yang and Jiaan Wang and Chunping Wang and Yang Yang},
  booktitle={Proceedings of the 31th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```
