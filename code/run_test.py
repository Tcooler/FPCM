import os
from test import test
import torch
import pandas as pd
import numpy as np

def check_fasta_file(file_path):
    if not os.path.isfile(file_path):
        raise Exception(f"Error: \'{file_path}\' does not exist")
    
    # 检查文件扩展名是否是 .fasta 或 .fa
    if not file_path.endswith(('.fasta', '.fa')):
        raise Exception(f"'{file_path}'is not a fasta file")

def check_for_unstandard_amino_acids(sequence):
    standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    for amino_acid in sequence:
        if amino_acid not in standard_amino_acids:
            return False
    return True

def read_data(pos_file,neg_file):
    P_File = open(pos_file,'r')
    P_Sequences = []
    for line in P_File.readlines():
        if not line.startswith('>'):
            sequence = line.rstrip()
            if check_for_unstandard_amino_acid(sequence):
                P_Sequences.append([1,sequence])
    P_File.close()
    N_File = open(neg_file,'r')
    N_Sequences = []
    for line in N_File.readlines():
        if not line.startswith('>'):
            sequence = line.rstrip()
            if check_for_unstandard_amino_acid(sequence):
                N_Sequences.append([0,sequence])
    N_File.close()
    test_data = P_Sequences + N_Sequences
    return test_data


parser = argparse.ArgumentParser()
parser.add_argument('--positive_samples', type=str, required=True, help="The fasta file of therapeutic peptide samples")
parser.add_argument('--negative_samples', type=str, required=True, help="The fasta file of non-therapeutic peptide samples")
parser.add_argument('--device', type=str, default='cpu', help="Please enter the device code, the default is cpu, enter for example \'cuda:0\'")
parser.add_argument('--pre_trained_model_dir',type=str,default='../protst_esm2_protein.pt',help="Pre-trained model locations to be fine-tuned")
parser.add_argument('--finetune_layernumber',type=int,default=10,help="Number of pre-trained model layers")
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
batch_size = args.batch_size
finetune_layernumber = args.finetune_layernumber
device = torch.device(device if torch.cuda.is_available() else "cpu")

test_data = train_data,test_data = read_data(positive_file,negative_file)
result = test(test_data,model_file=model_file,device=device,pre_trained_model_dir=pre_trained_model_dir,batch_size=batch_size,min_layernumber=min_layernumber)
data_column = ['accuracy','auc','f1','tn','fp','fn','tp']
data_record = pd.DataFrame([data_result],columns=data_column)
if os.path.exists(result_file):
    data_record_ori = pd.read_csv(result_file)
    data_record = pd.concat([data_record_ori,data_record],ignore_index=True)
data_record.to_csv(result_file,index=False)
