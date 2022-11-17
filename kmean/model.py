# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import logging

import sys
sys.path.append("..")
from train.util import CfgNode as CN
from normal import Normal

logger = logging.getLogger(__name__)

class Kmodel(nn.Module):
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.dead_cluster = 3
        C.useVar = True
        C.ncluster = 512
        C.ndim = 8
        C.dataset = None        
        C.loadPara = True        
        C.initParaRange = 2048        
        C.dl_mode = True        
        return C
    
    
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        if  config.loadPara:
            self.mu = nn.Parameter(torch.zeros(self.cfg.ncluster, 
                               self.cfg.ndim))
            self.var = None                    
            if config.useVar:                   
                self.var = nn.Parameter(torch.zeros(self.cfg.ncluster, 
                                        self.cfg.ndim, self.cfg.ndim))
                                           
        else:            
            
            assert self.cfg.dataset is not None
            print(len(self.cfg.dataset))          
            self.mu, self.var = Normal.getInitPara(self.cfg.dataset, 
                                          self.cfg.ncluster, 
                                          self.cfg.initParaRange,
                                          self.cfg.useVar)
                                          
            self.mu = nn.Parameter(self.mu)
            if self.var is not None:
                self.var = nn.Parameter(self.var)
        
        if not self.cfg.dl_mode:
            self.idx_list = torch.empty((0,))
            
        '''
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                print(pn, p, p.shape)
        '''
        
        
    def param(self):
        return self.mu.cpu().detach(), None if self.var is None else self.var.cpu().detach()    
        
    def update(self):
        if self.cfg.dl_mode: return               
        
        mu, var = Normal.M_step(self.cfg.dataset, self.idx_list,
                                self.cfg.ncluster, self.cfg.useVar, 
                                self.cfg.dead_cluster)
        
        self.mu = nn.Parameter(mu.to(self.idx_list))
        if self.var is not None:
            self.var = nn.Parameter(var.to(self.idx_list))
        self.idx_list = torch.empty((0,))
     
    def config_optimizers(self, cfg):
        if not self.cfg.dl_mode: return None 
        #optimizer = torch.optim.SGD(params=self.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                      lr=cfg.learning_rate,
                                      weight_decay=0.0)
        return optimizer
        
        
    def forward(self, x):   
        
        idx, value = Normal.data2cluster(x, self.mu, self.var)        
        loss = value[:, 0].mean()
        
        if not self.cfg.dl_mode and self.training:
            self.idx_list = self.idx_list.to(self.mu)  
            self.idx_list = torch.cat([self.idx_list, idx[:, 0]], dim=0)  
        return idx, dict(total_loss=loss)    
       