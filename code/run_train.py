import os
from train import Train
import torch
import pandas as pd
import numpy as np
from read_data import read_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--positive_file', type=str, required=True, help="The fasta file of therapeutic peptide samples")
parser.add_argument('--negative_file', type=str, required=True, help="The fasta file of non-therapeutic peptide samples")
parser.add_argument('--train_num', type=int, default=20, help="How many positive and negative samples are used for training")
parser.add_argument('--test_num', type=int, default=-1, help="How many positive and negative samples are used for test")
parser.add_argument('--k', type=int, default=2, help="Number of clusters for clustering metric learning")
parser.add_argument('--seed', type=int, default=123, help="Random seed")
parser.add_argument('--device', type=str, default='cpu', help="Please enter the device code, the default is cpu, enter for example \'cuda:0\'")
parser.add_argument('--pre_trained_model_dir',type=str,default='../protst_esm2_protein.pt',help="Pre-trained model locations to be fine-tuned")
parser.add_argument('--model_dir',type=str,default='../model/',help='Where the model parameters are saved')
parser.add_argument('--result_file',type=str,default='../result.csv',help="The path of result file, please end with '.csv'")
parser.add_argument('--lr',type=float,default=0.006,help="Learning rate")
parser.add_argument('--finetune_layernumber',type=int,default=10,help="Number of pre-trained model layers")
parser.add_argument('--alpha',type=float,default=0.5,help="Hyperparameters used to balance CE loss and KMeans triplet loss")
parser.add_argument('--batch_size',type=int,default=20,help="Batch size")
parser.add_argument('--weight_decay',type=float,default=0.0001,help="Weight decay")
parser.add_argument('--epoch_num',type=int,default=50,help="Epoch number")


def write_result(result_file,result):
    OF = open(result_file,'a+')
    OF.write(result)
    OF.close()
    return
    
args = parser.parse_args()
positive_file = args.positive_file
negative_file = args.negative_file
train_num = args.train_num
test_num = args.test_num
k = args.k
seed = args.seed
device = args.device
weight_decay = args.weight_decay
if device.isdigit():
    device = 'cuda:'+device
pre_trained_model_dir = args.pre_trained_model_dir
model_dir = args.model_dir
result_file = args.result_file
lr = args.lr
finetune_layernumber = args.finetune_layernumber
alpha = args.alpha
if alpha<0 or alpha>1:
    raise Exception("The balance factor alpha must be between 0 and 1")
batch_size = args.batch_size
epoch_num = args.epoch_num

all_layernumber = 33 #Number of pre-trained model layers
if finetune_layernumber > 33:
    raise Exception("The number of layers to be fine-tuned cannot be greater than 33")
min_layernumber = all_layernumber - finetune_layernumber
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

train_data,test_data = read_data(positive_file,negative_file,train_num,test_num,seed)
model_file = model_dir + f'{seed}_{k}_{lr}_{min_layernumber}_{alpha}.pt'
result = Train(train_data,test_data,model_save_dir=model_file,device=device,pre_trained_model_dir=pre_trained_model_dir,lr=lr,mask_flag=True,alpha=alpha,min_layernumber=min_layernumber,loss_k=k,batch_size=batch_size,weight_decay=weight_decay)
data_result = [result['accuracy'],result['auc'],result['f1'],result['tn'],result['fp'],result['fn'],result['tp']]
data_result = [seed,k,lr,min_layernumber,alpha] + data_result
data_column = ['seed','k','lr','min_layernumber','alpha','accuracy','auc','f1','tn','fp','fn','tp']
data_record = pd.DataFrame([data_result],columns=data_column)
if os.path.exists(result_file):
    data_record_ori = pd.read_csv(result_file)
    data_record = pd.concat([data_record_ori,data_record],ignore_index=True)
data_record.to_csv(result_file,index=False)
