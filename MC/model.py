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
from mc import MC

logger = logging.getLogger(__name__)


class MCmodel(nn.Module):
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.ncluster = 64
        C.ndim = 32
        C.validset = None
        C.validsize = 2000
        C.parasize = 0    
        C.validmin = 50
        C.maxv = None
        C.minv = None        
        return C
    
    
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        #随机位置
        self.cluster = nn.Parameter(torch.zeros(self.cfg.ncluster, self.cfg.ndim)) 
        #随机权重
        self.weight =  nn.Parameter(torch.zeros(self.cfg.ncluster, self.cfg.ndim))
        #收益和阈值
        #0:测试收益率 1:选择到的数目 2:训练收益率, 3:阈值
        self.profit = nn.Parameter(torch.zeros(self.cfg.ncluster, 4))
                
        self.valuelist = torch.zeros(0)
        self.labellist = torch.zeros(0)
        
        '''
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                print(pn, p, p.shape)
        '''
    
        
    def param(self):
        return self.cluster.cpu().detach(), self.weight.cpu().detach(), \
               self.profit.cpu().detach()    
       
    
    def update(self):    
        
        va, idx = torch.sort(self.valuelist, dim=-1)        
        pos = torch.arange(idx.shape[0]).repeat(idx.shape[1], 1).T
        #print(pos.shape)
        label = self.labellist[pos, idx][:, :self.cfg.validsize]
        #训练阈值
        self.temp_profit[:, 3] = va[:, self.cfg.validsize-1]
        #训练收益率
        self.temp_profit[:, 2] = MC.profit(label)
        
        
        if self.cfg.validset is  not None:
            #验证, 0:收益律 1：分类数量
            self.temp_profit[:, :2] = MC.valid(self.cfg.validset, self.temp_cluster, 
                                 self.temp_weight,      self.temp_profit[:, 3])   
                               
        
        self.temp_cluster = torch.cat([self.cluster, self.temp_cluster], dim=0)
        self.temp_weight = torch.cat([self.weight, self.temp_weight], dim=0)
        self.temp_profit = torch.cat([self.profit, self.temp_profit], dim=0)
        
        #要求足够数量的验证
        cond = self.temp_profit[:, 1] > self.cfg.validmin
        self.temp_cluster = self.temp_cluster[cond]
        self.temp_weight = self.temp_weight[cond]
        self.temp_profit = self.temp_profit[cond]
        
        # 按照收益率排序
        v = self.temp_profit[:, [0, 2]].min(dim=1)[0] *0.8 +\
            self.temp_profit[:, [0, 2]].max(dim=1)[0]*0.2
        
        va, idx = torch.sort(v, dim=-1, descending=True)
        
        self.cluster = nn.Parameter(self.temp_cluster[idx[:self.cfg.ncluster]])
        self.weight = nn.Parameter(self.temp_weight[idx[:self.cfg.ncluster]])
        self.profit = nn.Parameter(self.temp_profit[idx[:self.cfg.ncluster]])
        
        print(self.profit[:10], self.profit.shape)
        self.valuelist = torch.zeros(0)
        self.labellist = torch.zeros(0)
        
    
    def config_optimizers(self, cfg):
        return None 
        
    
    def forward(self, x):    
    
        data, label = x        
        data = data.to(self.cluster)
        label = label.to(self.cluster) if label is not None else None
        
        if self.training:
            self.temp_cluster, self.temp_weight =   MC.getZoneRnd((self.cfg.parasize,
                           self.cfg.ndim), self.cfg.minv, self.cfg.maxv)      
            self.temp_profit = torch.zeros(self.cfg.parasize,  4)
            
            self.temp_cluster = self.temp_cluster.to(data)
            self.temp_weight =  self.temp_weight.to(data)
            self.temp_profit = self.temp_profit.to(data)
            
            va, idx = MC.data2cluster(data, self.temp_cluster, self.temp_weight)   
            
            va = va[:, :self.cfg.validsize]            
            idx = idx[:, :self.cfg.validsize]     
            
            self.valuelist, self.labellist =  self.valuelist.to(data), self.labellist.to(data)            
            self.valuelist = torch.cat((self.valuelist, va), dim=-1)
            self.labellist = torch.cat((self.labellist, label[idx]), dim=-1)
            
            loss = MC.profit(label[idx])     
            
            return idx, dict(total_loss=loss.max())
        else:
            va, idx = MC.data2cluster(data, self.cluster, self.weight)
            pos = MC.selectwiththd(va, idx, self.profit[:, 3])
            
            if label is not None:
                stat = []
                for one in pos:
                    stat.append([MC.profit(label[one][None]), len(one)])
                return pos, torch.tensor(stat)
            return pos, None  
                    
        