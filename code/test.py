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
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score, confusion_matrix,silhouette_score

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
    
def deal_with_y(labels):
    Y = []
    for label in labels:
        y = int(label)
        Y.append(y)
    Y = torch.tensor(Y)
    return Y

def test(test_data,model_file,device,pre_trained_model_dir='../protst_esm2_protein.pt',esm_model_dir='../protst_esm2_protein.pt',batch_size=20,min_layernumber=10):
    print(f'number of samples for test:{len(test_data)}')
    test_data = data.DataLoader(test_data,batch_size,shuffle=True)
    model = Model(min_layernumber=min_layernumber)
    model = add_param(model,model_file=model_file,esm_model_dir=pre_trained_model_dir)
    for name,param in model.named_parameters():
        param.requires_grad = False
    model = model.to(device)
    Y_classes = []
    Y_class_hats = []
    Y_class_hat_scores = []
    for labels,X in tes_data:
        Y_class = deal_with_y(labels).to(device)
        Y_class_hat,_ = model(labels,X,device)
        probability = F.softmax(Y_class_hat,dim=1)
        Y_class_hat_scores.append(probability[:, 1].cpu().detach().numpy())
        Y_classes.append(Y_class.cpu().detach().numpy())
        Y_class_hats.append(Y_class_hat.argmax(axis=1).cpu().detach().numpy())
    Y_classes = np.concatenate(Y_classes,0)
    Y_class_hats = np.concatenate(Y_class_hats,0)
    Y_class_hat_scores = np.concatenate(Y_class_hat_scores)
    accuracy = accuracy_score(Y_classes, Y_class_hats)
    auc = roc_auc_score(Y_classes,Y_class_hat_scores)
    f1 = f1_score(Y_classes, Y_class_hats, average='binary')
    cm = confusion_matrix(Y_classes, Y_class_hats)
    tn,fp,fn,tp = cm[0][0]*2/len(test_data),cm[0][1]*2/len(test_data),cm[1][0]*2/len(test_data),cm[1][1]*2/len(test_data)
    print(f'Test :ACC score: {accuracy}  AUC score: {auc}')
    result = {'accuracy':accuracy,'auc':auc,'f1':f1,'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    return result
