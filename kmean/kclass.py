import math
import numpy as np
import pandas as pd
import torch
import random
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append("..")
from train.util import set_seed
from train.util import CfgNode as CN
from normal import Normal
import time


class Kclass:
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.mu = None
        C.var = None
        C.anspath = './data/ans.pkl'
        return C
        

    def __init__(self, config):
        self.cfg =  config
        
    def create_class(self, dataset):
        
        bs = 40960 
        mu, var = self.cfg.mu, self.cfg.var
        info = dataset.info()
        loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                            batch_size=bs, num_workers=0)
        
        idxlist= []
        problist = []
        for it, x in enumerate(loader):            
            idx, value =  Normal.data2cluster(x, mu, var) 
            prob = Normal.prob(value[:, :16])
            idxlist.append(idx[:, :5])
            problist.append(prob[:, :5])
        
        idxlist = torch.cat(idxlist, dim=0)
        problist = torch.cat(problist, dim=0)
        ret = pd.DataFrame()
        ret['code'] = info[:, 0]
        ret['day'] = info[:, 1].astype(int)
        ret['ret'] = info[:, 2].astype(int)
        ret['idx_0'] = idxlist[:, 0].numpy()        
        ret['value_0'] = problist[:, 0].numpy()
        ret['idx_1'] = idxlist[:, 1].numpy()        
        ret['value_1'] = problist[:, 1].numpy()
        ret['idx_2'] = idxlist[:, 2].numpy()        
        ret['value_2'] = problist[:, 2].numpy()
        ret['idx_3'] = idxlist[:, 3].numpy()        
        ret['value_3'] = problist[:, 3].numpy()
        ret['idx_4'] = idxlist[:, 4].numpy()        
        ret['value_4'] = problist[:, 4].numpy()        
        ret.to_pickle(self.cfg.anspath)
    
    def findwithdata(self, data):
        if len(data.shape) == 1:
            data = data[None]            
        mu, var = self.cfg.mu, self.cfg.var
        idx, value =  Normal.data2cluster(data, mu, var)
        idx = idx[0,:5].tolist()        
        return self.findwithidx(idx[0], set(idx[1:]))        
        
            
    def findwithcode(self, code, day):
        ans = pd.read_pickle(self.cfg.anspath)
        return ans[(ans.code==code)&(ans.day==day)]
    
    def findwithidx(self, idx_0, idxset=None):
        ans = pd.read_pickle(self.cfg.anspath)
        ans = ans[ans.idx_0==idx_0]
        
        isPos = lambda x: [(x.iloc[i].idx_1 in idxset) & (x.iloc[i].idx_2 in idxset) &\
                           (x.iloc[i].idx_3 in idxset) & (x.iloc[i].idx_4 in idxset)\
                           for i in range(x.shape[0]) ]
        if idxset is not None:
            pos = isPos(ans)
            return ans[pos]
        return ans            
    
if __name__ == '__main__':
    pass