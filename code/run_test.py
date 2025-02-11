import os
from Test import Test
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=str, required=True, help="The fasta file of peptides for test")
parser.add_argument('--device', type=str, default='cpu', help="Please enter the device code, the default is cpu, enter for example \'cuda:0\'")
parser.add_argument('--pre_trained_model_dir',type=str,default='../protst_esm2_protein.pt',help="Pre-trained model locations to be fine-tuned")
parser.add_argument('--model_file',type=str,default='../model/',help='Where the model parameters are saved')
parser.add_argument('--result_file',type=str,default='../test_result.csv',help="The path of result file, please end with '.csv'")
parser.add_argument('--batch_size',type=int,default=20,help="Batch size")
    
args = parser.parse_args()
samples_file = args.samples
device = args.device
if device.isdigit():
    device = 'cuda:'+device
pre_trained_model_dir = args.pre_trained_model_dir
model_file = args.model_file
result_file = args.result_file
if alpha<0 or alpha>1:
    raise Exception("The balance factor alpha must be between 0 and 1")
batch_size = args.batch_size
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

test_data = read_test_data(samples_file)
result = Test(test_data,model_file=model_file,device=device,pre_trained_model_dir=pre_trained_model_dir,batch_size=batch_size)
data_result = [result['sequence'],result['flag'],result['score']]
data_column = ['sequence','flag','score']
data_record = pd.DataFrame([data_result],columns=data_column)
if os.path.exists(result_file):
    data_record_ori = pd.read_csv(result_file)
    data_record = pd.concat([data_record_ori,data_record],ignore_index=True)
data_record.to_csv(result_file,index=False)
