import torch
import numpy as np
import re
import os
import random
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans


class FewSOME(nn.Module):
    def __init__(self):
        super(FewSOME, self).__init__()
    def forward(self,sequence_representations_p,sequence_representations_n):
        l_p = len(sequence_representations_p)
        l_n = len(sequence_representations_n)
        sequence_representations_p = torch.nn.functional.normalize(sequence_representations_p, p=2, dim=1)
        sequence_representations_n = torch.nn.functional.normalize(sequence_representations_n, p=2, dim=1)
        distance = torch.zeros(l_p,l_p)
        t = random.randint(0,l_n-1)
        for i in range(l_p):
            for j in range(l_p):
                d_pp = F.pairwise_distance(sequence_representations_p[i],sequence_representations_p[j])
                d_pn = F.pairwise_distance(sequence_representations_p[i],sequence_representations_n[t])
                distance[i][j] = max(0,d_pp + 0.2 - d_pn)
        distance = distance.to(sequence_representations_p.device)
        distance_loss = distance.sum(dim=1)/((l_p-1)*l_p)
        #distance_loss = 0.5*(distance)**2
        distance_loss = distance_loss.sum()
        return max(0,distance_loss)

class Loss(nn.Module):
    def __init__(self,k,alpha=2):
        super(Loss, self).__init__()
        self.k = k
        self.fewsome = FewSOME()
        self.class_loss_f = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = alpha
    def forward(self,sequence_representaitions,y_hat,y):
        class_loss = self.class_loss_f(y_hat,y)
        kmeans = KMeans(n_clusters=self.k,random_state=0)
        kmeans.fit(sequence_representaitions.cpu().detach().numpy())
        labels = kmeans.labels_
        distance_loss = 0
        for i in range(self.k):
            num_list_p = []
            num_list_n = []
            for j,label in enumerate(labels):
                if label == i:
                    if y[j] == 1:
                        num_list_p.append(j)
                    else:
                        num_list_n.append(j)
            samples_p = sequence_representaitions[num_list_p]
            samples_n = sequence_representaitions[num_list_n]
            if len(num_list_p) <= 1 or len(num_list_n) <= 0:
                continue
            distance_loss += self.fewsome(samples_p,samples_n)
        print(class_loss,distance_loss)
        loss = class_loss + self.alpha*distance_loss
        return loss
