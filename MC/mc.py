import math
import numpy as np
import pandas as pd
import torch
import random
import os
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append("..")

import time
import logging
logger = logging.getLogger(__name__)


class MC:

    def __init__(self):
        pass
    
    
    @classmethod
    def getZoneRnd(cls, size, mi, ma):
        temp = torch.rand(size)        
        #(0,1)映射到(mi,ma)区间: mi +  (ma - mi) * (temp - 0) / (1- 0)
        temp = temp * (ma - mi) +  mi        
        prob = 10 * torch.rand(size)
        #temp = temp / temp.sum(dim=0)
        prob = F.softmax(prob, dim=-1)                
        return temp, prob     
    
    @classmethod    
    def data2cluster(cls, data, clu, prob):
        #计算欧氏距离
        m = data[:, None, :] - clu[None]   
        m = m ** 2
        #维度乘以概率
        m = m * prob[None]
        m = m.sum(dim=-1).T
        va, idx = torch.sort(m, dim=-1)
        return va, idx  
    
    @classmethod
    def profit(cls, label):
        label = label == 1                
        return label.sum(dim=-1) / torch.clamp(torch.tensor(label.shape[1], dtype=torch.float32), 
                                         min=torch.finfo(torch.float32).eps) 
        
        profit = []
        for k in range(label.shape[0]):            
            w = label[k]
            profit.append(w[w==1].shape[0] / w.shape[0])            
        profit = torch.tensor(profit)
        return profit
    
    @classmethod
    def selectwiththd(cls, va, idx, threshold):
        temp = []
        for k in range(threshold.shape[0]):
            id = idx[k, va[k] < threshold[k]]
            temp.append(id.tolist())
        return temp
        
    
    @classmethod
    def valid(cls, dataset, clu, weight, threshold):
        
        bs = 40960
        loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                            batch_size=bs, num_workers=0)
        
        temp = [[] for i in range(clu.shape[0])]
        
        for it, (x, l) in enumerate(loader):
            x = x.to(clu)
            l = l.to(clu)            
            va, idx = cls.data2cluster(x, clu, weight)
            ans = cls.selectwiththd(va, idx, threshold)
            temp = [ temp[i] + l[ans[i]].tolist() for i in range(len(ans))]
            
        stat = []
        for one in temp:
            one = torch.tensor(one)            
            stat.append([MC.profit(one[None]), one.shape[0]])        
        
        return torch.tensor(stat)
       
        
        
        