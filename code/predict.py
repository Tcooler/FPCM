import torch
import numpy as np
import re
import os
import random
from torch.utils import data
import torch.nn as nn
from protein_model import Model
import torch.nn.functional as F
from Loss import Loss

def add_param(model,model_file=None,esm_model_dir='../protst_esm2_protein.pt'):
    #esm_model_dir = '/data3/yuanxilu/thrpep_new/model/esm2_t33_650M_UR50D.pt'
    print(f'Start loading model')
    pretrained_dict = torch.load(esm_model_dir) 
    model_dict = model.state_dict()
    for name, param in pretrained_dict.items():
        name_now = 'esm.'+name[20:]
        #name_now = name[25:]
        if name_now in model_dict:
            model_dict[name_now].copy_(param)
    if model_file:
        trained_dict = torch.load(model_file)
        for name, param in trained_dict.items():
            if name in model_dict:
                model_dict[name].copy_(param)
    model.load_state_dict(model_dict,strict=False)
    for name,param in model.named_parameters():
        param.requires_grad = False
    print(f'Finish loading model')
    return model


def predict(input_data,model_file,device,pre_trained_model_dir='../protst_esm2_protein.pt',esm_model_dir='../protst_esm2_protein.pt',batch_size=20,min_layernumber=10):
    print(f'number of samples for predict:{len(input_data)}')
    input_data = data.DataLoader(input_data,batch_size,shuffle=False)
    model = Model(min_layernumber=min_layernumber)
    model = add_param(model,model_file=model_file,esm_model_dir=pre_trained_model_dir)
    for name,param in model.named_parameters():
        param.requires_grad = False
    model = model.to(device)
    Y_class_hats = []
    Y_class_hat_scores = []
    model.eval()
    for labels,X in input_data:
        Y_class_hat,sequence_represent = model(labels,X,device,mask_flag=False)
        probability = F.softmax(Y_class_hat,dim=1)
        Y_class_hat_scores.append(probability[:, 1].cpu().detach().numpy())
        Y_class_hats.append(Y_class_hat.argmax(axis=1).cpu().detach().numpy())
        Y_class_hats = np.concatenate(Y_class_hats,0)
        Y_class_hat_scores = np.concatenate(Y_class_hat_scores)
    return Y_class_hats,Y_class_hat_scores
