# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import logging

from torch.distributions.multivariate_normal import  MultivariateNormal
from scipy.stats import multivariate_normal
import sys
sys.path.append("..")
from train.util import CfgNode as CN
from train.util import np2torch

from kmean.normal import Normal
from gmm import Gmm_np as GM
#from gmm import Gmm_torch as GM

logger = logging.getLogger(__name__)

class Gmm(nn.Module):
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.ncluster = 512
        C.ndim = 8
        C.dataset = None        
        C.dl_mode = False
        C.initParaRange = 2048
        C.loadPara = False
        return C
    
    
    def __init__(self, config):
        super().__init__()
        self.cfg = config
         
        
        if  config.loadPara:
            self.mu = nn.Parameter(torch.zeros(self.cfg.ncluster, 
                               self.cfg.ndim))
            self.var = nn.Parameter(torch.zeros(self.cfg.ncluster, 
                                        self.cfg.ndim, self.cfg.ndim))
                                           
        else:            
            assert self.cfg.dataset, 'need dataset'
            print(len(self.cfg.dataset))         
            self.mu, self.var = Normal.getInitPara(self.cfg.dataset, 
                                               self.cfg.ncluster, 
                                               self.cfg.initParaRange,
                                               True)
                                          
        self.mu = nn.Parameter(self.mu)
        self.var = nn.Parameter(self.var)
        self.Pi = nn.Parameter(torch.ones(self.mu.shape[0]) / self.mu.shape[0])  
        
        logger.info('%s, %s' %(self.var.shape, self.mu.shape))
        
        '''            
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                print(pn, p, p.shape)
        '''
    
    def param(self):
        return self.mu.cpu().detach(), self.var.cpu().detach(),\
               self.Pi.cpu().detach()    
        
    def update(self):
        if self.cfg.dl_mode: return 
    
    def config_optimizers(self, cfg):
        if not self.cfg.dl_mode: return None 
        #optimizer = torch.optim.SGD(params=self.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                      lr=cfg.learning_rate,
                                      weight_decay=0.0)
        return optimizer
    
    def forward(self, x):    
        
        x = x.to(self.mu)
        W = GM.update_W(x, self.mu, self.var, self.Pi)
                
        loss = None
        if  self.training:
            Pi = GM.update_Pi(W)                 
            mu = GM.update_Mu(x, W)      
            var = GM.update_Var(x, W)
        
            loss = GM.logLH(x, Pi, mu, var)      
            self.mu = nn.Parameter(np2torch(mu).to(self.mu))
            self.var = nn.Parameter(np2torch(var).to(self.mu))
            self.Pi = nn.Parameter(np2torch(Pi).to(self.mu))
            loss = dict(total_loss=np2torch(loss))
            
        return np2torch(W), loss    
       
       
       
       
       