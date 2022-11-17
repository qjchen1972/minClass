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
import time

from model import MCmodel
from train.trainer import Trainer
from train.util import set_seed
    

class StockDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        #数据格式是 n*10
        self.data = np.load(path,allow_pickle=True)        
        #self.a_m = [0.4938,0.6084,0.3834,0.4914,1.0037,0.0308,1.0094,0.0311,1.0097,0.0318]
        #self.a_s = [0.0750,0.1271,0.1165,0.1658,0.3649,0.0460,0.4921,0.0454,0.6284,0.0450]
        self.a_m = [0.4942, 0.4942, 0.4942, 0.4942, 1.0076, 0.0312, 1.0076, 0.0312, 1.0076, 0.0312]
        self.a_s = [0.1485, 0.1485, 0.1485, 0.1485, 0.5067, 0.0455, 0.5067, 0.0455, 0.5067, 0.0455]
        self.a_m = torch.tensor(self.a_m)
        self.a_s = torch.tensor(self.a_s)
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        # 读出来的数据是strt类型, 需要转换
        temp = torch.from_numpy(self.data[idx, 3:13].astype(np.float32))
        temp = (temp - self.a_m) / torch.clamp(self.a_s, min=torch.finfo(torch.float32).eps)
        return temp, self.data[idx, 2].astype(int)
        #torch.from_numpy(self.data[idx, 3:].astype(np.float32))
    
    def info(self, idx=None):
        
        return self.data[:,:3] if idx is None else  self.data[idx,:3]
    
    def getitem(self, code, day):
        pos = np.where((self.data[:, 0]==code)&(self.data[:, 1]==str(day)))
        return self.__getitem__(pos[0][0])
    

    
def minMax(datset):
    loader = DataLoader(datset, pin_memory=False, shuffle=False,
                        batch_size=len(datset), num_workers=0) 
    data, _ = next(iter(loader))
    ma = data.max(dim=0)
    mi = data.min(dim=0)
    return ma[0], mi[0]
    
def train(train_dataset, valid_dataset, lastmodel=None):

    ma, mi = minMax(train_dataset)
    model_config = MCmodel.get_default_config()
    model_config.ncluster = 64
    model_config.ndim = 10
    model_config.validsize = 2000
    model_config.validset = valid_dataset
    model_config.parasize = 2048
    model_config.maxv = ma
    model_config.minv = mi
    model = MCmodel(model_config)
    
    
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-3 
    train_config.warmup_tokens = 0
    train_config.max_epochs = 20000
    train_config.num_workers = 0
    #train_config.shuffle = True
    train_config.batch_size = 40960 #len(train_dataset)
    train_config.distributed = False
    train_config.rndseed = 7612
    #train_config.device = 'cpu'
    trainer = Trainer(train_config, model, train_dataset)
    if lastmodel is not None:
        trainer.load_lastpoint(lastmodel)        
    trainer.train()            

def test(dataset, lastmodel):
    model_config = MCmodel.get_default_config()
    model_config.ncluster = 64
    model_config.ndim = 10
    model = MCmodel(model_config)
    
    train_config = Trainer.get_default_config()
    train_config.distributed = False
    #train_config.device = 'cpu'
    trainer = Trainer(train_config, model, train_dataset=None)
    model = trainer.load_lastpoint(lastmodel)  
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()    
    
    loader = DataLoader(dataset, pin_memory=True, shuffle=False,
                        batch_size=len(dataset), num_workers=0)
    data, label = next(iter(loader))
    
    idx, profit = model((data, label))
    num = 62
    t = dataset.info(idx[num])
    print(t[:30], t.shape, profit[num])
    
    
    
if __name__ == '__main__':
    '''
    torch.set_printoptions(
    precision=4,    # 精度，保留小数点后几位，默认4
    threshold=float('inf'),
    edgeitems=3,
    #linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    #sci_mode=False  # 用科学技术法显示数据，默认True
    )
    '''
    
    tr_path = '../data/MC/train.npy'
    te_path = '../data/MC/test.npy'
    va_path = '../data/MC/valid.npy'
    
    tr = StockDataset(tr_path)
    va = StockDataset(va_path)
    
    #set_seed(79)    
    train(tr, va, 2468)
    
    #te = StockDataset(te_path)
    #test(te, 2468)
    