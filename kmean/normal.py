import math
import numpy as np
import pandas as pd
import torch
import random
import os
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions.multivariate_normal import  MultivariateNormal

import sys
sys.path.append("..")

from train.util import set_seed
import time

import logging
logger = logging.getLogger(__name__)


class Normal:

    diag = False
    niter = 0
    
    def __init__(self):
        pass
    
    @classmethod    
    def getVar(cls, dataset, mu):
        loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                            batch_size=len(dataset), num_workers=0)
        
        data = next(iter(loader))
        idx, value = cls.data2cluster(data, mu)        
        para = []
        for k in range(mu.shape[0]):
            x = data[idx[:, 0]==k]
            x = x - mu[k][None]             
            var = torch.mm(x.T, x) / (x.shape[0] - 1)
            if torch.any(torch.isinf(var)):
                var += float('nan')                
            if cls.diag:
                var = torch.clamp(torch.diag(var), min=torch.finfo(torch.float32).eps)
                var = torch.diag_embed(var)
            para.append(var)            
        para = torch.stack(para, dim=0)    
        return para
    
    
    @classmethod    
    def prob(cls, data):
        
        #torch.set_printoptions(edgeitems=8, )
        if data.shape[1] == 1:
            prob = torch.zeros(data.shape) + 1
            prob = prob.to(data)
        else:        
            
            norm = (data - data.mean(dim=1)[:, None]) /\
                torch.clamp(data.std(dim=1)[:, None], min=torch.finfo(torch.float32).eps)  
            #这种标准化效果不好        
            #norm = F.normalize(value)
            norm = -norm
            prob = torch.nn.Softmax(dim=1)(norm)
        return prob        
        
        
    
    @classmethod    
    def getMu(cls, dataset, ncluster, idxlist, ndead=0):
    
        loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                            batch_size=len(dataset), num_workers=0)
        
        data = next(iter(loader))
        mu = []         
        for k in range(ncluster):            
            t = data[idxlist==k]
            if t.shape[0] <= ndead:
                t = torch.empty((0, data.shape[1]))                
            mu.append(t.mean(dim=0))        
        mu = torch.stack(mu, dim=0)
        return mu
    
    @classmethod    
    def getValidPara(cls, dataset, mu):
        
        var = cls.getVar(dataset, mu)   
        for i in range(1, cls.niter):
            nanix = torch.tensor([torch.any(torch.isnan(var[j])) or \
                                  torch.any(torch.isinf(var[j])) \
                                 for j in  range(var.shape[0])])
                                       
            ndead = nanix.sum().item()  
            if ndead == 0: break
            logger.info('iter %d, normal re-initialized %d dead clusters' % (i, ndead))
            mu[nanix] = cls.getInitMu(dataset, ndead, existmu=mu[~nanix])
            var = cls.getVar(dataset, mu) 
        return mu, var
    
        
    @classmethod    
    def euler(cls, x, c):
        temp = x[:, None, :] - c[None, :, :]
        temp = temp ** 2
        temp = temp.sum(dim=-1)
        #value, idx = temp.min(dim=1)
        v, i = temp.sort(dim=1, descending=False)
        return i, v
    
    @classmethod
    def gas(cls, x, mu, var):
        temp = []
        for k in range(mu.shape[0]):
            try:
                normal = MultivariateNormal(mu[k], var[k])
                temp.append(-normal.log_prob(x))                
            except ValueError as e:
                temp.append(torch.zeros(x.shape[0]).to(x) + float('inf')) 
              
        temp = torch.stack(temp, dim=0).T  
        v, i = temp.sort(dim=1, descending=False)
        #value, idx = temp.min(dim=1)        
        return i, v
    
    @classmethod    
    def data2cluster(cls, x, mu, var=None):
        if isinstance(x, list):
            x = torch.from_numpy(np.array(x))
        elif isinstance(x, np.ndarray):    
            x = torch.from_numpy(x) 
        x = x.to(mu)   
        
        if var is None:
            return cls.euler(x, mu)
        else:
            return cls.gas(x, mu, var)
            
    @classmethod    
    def getInitMu(cls, dataset, num, bs=2048, existmu=None):  
      
        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=bs, num_workers=0)        
        
        #data = next(iter(loader))
        #return data[torch.randperm(bs)[:num]]
        
        mu = None
        begin = 0
        if existmu is  not None:
            mu = existmu.cpu().detach()        
            begin = existmu.shape[0]
        
        for i in range(num):         
            data = next(iter(loader))
            if mu is None:
                mu = torch.stack([data[0]], dim=0)                
            else:                
                idx, va = cls.data2cluster(data, mu)                
                va = va[:, 0:1]
                mu = torch.concat([mu, data[va.max(dim=0)[1]]], dim=0)
        return mu[begin:] 
        
    @classmethod    
    def getInitPara(cls, dataset, num, bs, useVar=False): 
        
        mu = cls.getInitMu(dataset, num, bs)
        if useVar:
            return  cls.getValidPara(dataset, mu)
        else:
            return mu, None        

                        
    @classmethod 
    def M_step(cls, dataset, idxlist, ncluster, useVar, ndead=0):
        
        mu = cls.getMu(dataset, ncluster, idxlist, ndead)
        
        nanix =  torch.any(torch.isnan(mu), dim=1)
        ndead = nanix.sum().item()            
        logger.info('mean re-initialized %d dead clusters' % (ndead))
        if ndead > 0:            
            mu[nanix] = cls.getInitMu(dataset, ndead, existmu=mu[~nanix])
        if  useVar:
            return cls.getValidPara(dataset, mu)
        return mu, None
    