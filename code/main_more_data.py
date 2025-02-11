import os
from Train import Train
import torch
import pandas as pd
import numpy as np
from read_data_more_data import read_data

def write_result(result_file,result):
    OF = open(result_file,'a+')
    OF.write(result)
    OF.close()
    return

seeds = [123,456,789,101112,131415,161718,192021,222324,252627,282930]
n_tra_s = [10,20,50,100,200,500,1000]
n_tes = 500
lr = 0.003
min_layernumber = 23
alpha = 1
k = 2
torch.cuda.set_device(3)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_dir = '../model_more_data/'
result_file = '../result_more_data.csv'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
for dataset in ['AFP','AMP']:
    positive_file = '../dataset/'+dataset+'/positive.fasta'
    negative_file = '../dataset/'+dataset+'/negative.fasta'
    for n_tra in n_tra_s:
        if dataset == 'AFP' and n_tra == 1000:
            continue
        for seed in seeds:
            train_data,test_data = read_data(positive_file,negative_file,seed,n_tra,n_tes)
            model_file = model_dir + f'{dataset}_{seed}_{k}_{lr}_{min_layernumber}_{alpha}_{n_tra}_{n_tes}.pt'
            result = Train(train_data,test_data,model_save_dir=model_file,device=device,lr=lr,mask_flag=True,alpha=alpha,min_layernumber=min_layernumber,loss_k=k)
            data_result = [result['accuracy'],result['auc'],result['f1'],result['tn'],result['fp'],result['fn'],result['tp']]
            data_result = [dataset,n_tra*2,n_tes*2,seed,k,lr,min_layernumber,alpha] + data_result
            data_column = ['dataset','train_number','test_number','seed','k','lr','min_layernumber','alpha','accuracy','auc','f1','tn','fp','fn','tp']
            data_record = pd.DataFrame([data_result],columns=data_column)
            if os.path.exists(result_file):
                data_record_ori = pd.read_csv(result_file)
                data_record = pd.concat([data_record_ori,data_record],ignore_index=True)
            data_record.to_csv(result_file,index=False)

