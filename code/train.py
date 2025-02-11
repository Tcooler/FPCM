import torch
import numpy as np
import re
import os
import random
from torch.utils import data
import torch.nn as nn
from protein_model import Model
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score, confusion_matrix,silhouette_score
import torch.nn.functional as F
from Loss import Loss
from sklearn.cluster import KMeans


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def deal_with_y(labels):
    Y = []
    for label in labels:
        y = int(label)
        Y.append(y)
    Y = torch.tensor(Y)
    return Y


def add_param(model,model_dir=None,esm_model_dir='../protst_esm2_protein.pt'):
    #esm_model_dir = '/data3/yuanxilu/thrpep_new/model/esm2_t33_650M_UR50D.pt'
    print(f'Start loading model')
    pretrained_dict = torch.load(esm_model_dir) 
    model_dict = model.state_dict()
    for name, param in pretrained_dict.items():
        name_now = 'esm.'+name[20:]
        #name_now = name[25:]
        if name_now in model_dict:
            model_dict[name_now].copy_(param)
    if model_dir:
        trained_dict = torch.load(model_dir)
        for name, param in trained_dict.items():
            if name in model_dict:
                model_dict[name].copy_(param)
    model.load_state_dict(model_dict,strict=False)
    for name,param in model.named_parameters():
        param.requires_grad = False
    print(f'Finish loading model')
    return model


def Train(train_data,test_data,model_save_dir,device,pre_trained_model_dir='../protst_esm2_protein.pt',loss_k=2,mask_flag=False,seed=10,esm_model_dir='../protst_esm2_protein.pt',batch_size=20,lr=0.006,epoch_num=50,min_layernumber=23,weight_decay=0.0001,alpha=0.5):
    setup_seed(seed)
    print(len(train_data),len(test_data))
    tra_data = data.DataLoader(train_data,batch_size,shuffle=True)
    model = Model(min_layernumber=min_layernumber)
    model = add_param(model,esm_model_dir=pre_trained_model_dir)
    for name,param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        elif 'esm' not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model = model.to(device)
    loss_function = Loss(k=loss_k,alpha=alpha)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)
    result = {'accuracy':0,'auc':0,'f1':0,'tn':0,'fp':1,'fn':1,'tp':0}
    for epoch in range(epoch_num):
        Y_classes = []
        Y_class_hats = []
        Y_class_hat_scores = []
        model.train()
        for labels,X in tra_data:
            Y_class = deal_with_y(labels).to(device)
            Y_class_hat,sequence_represent = model(labels,X,device,mask_flag=mask_flag)
            if mask_flag:
                Y_class = torch.cat([Y_class,Y_class],0)
            loss = loss_function(sequence_represent,Y_class_hat,Y_class)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probability = F.softmax(Y_class_hat,dim=1)
            Y_class_hat_scores.append(probability[:, 1].cpu().detach().numpy())
            Y_classes.append(Y_class.cpu().detach().numpy())
            Y_class_hats.append(Y_class_hat.argmax(axis=1).cpu().detach().numpy())
        Y_classes = np.concatenate(Y_classes,0)
        Y_class_hats = np.concatenate(Y_class_hats,0)
        Y_class_hat_scores = np.concatenate(Y_class_hat_scores)
        accuracy = accuracy_score(Y_classes, Y_class_hats)
        auc = roc_auc_score(Y_classes,Y_class_hat_scores)
        print(f'Train epoch {epoch}:ACC score: {accuracy}  AUC score: {auc}')
        result_now = Test(model,test_data,device)
        if result_now['accuracy'] > result['accuracy'] or (result_now['accuracy'] == result['accuracy'] and result_now['auc'] > result['auc']):
            result = result_now
            save(model,model_save_dir)
    return result


def Test(model,test_data,device,batch_size=40):
    tes_data = data.DataLoader(test_data,batch_size,shuffle=False)
    model = model.to(device)
    model.eval()
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

def get_represent(X_data,model,device,batch_size=20):
    X_data = data.DataLoader(X_data,batch_size,shuffle=False)
    model = model.to(device)
    model.eval()
    sequence_respresents = []
    for labels,X in X_data:
        _,sequence_represent = model(labels,X,device)
        sequence_respresents.append(sequence_represent)
    sequence_respresents = torch.cat(sequence_respresents,0)
    return sequence_respresents


def save(model,model_save_dir):
    state_dict = {}
    for name,param in model.state_dict().items():
        if 'lora' in name:
            state_dict[name]=param
        elif 'esm' not in name:
            state_dict[name]=param
    torch.save(state_dict, model_save_dir)
