# -*- coding:utf-8 -*-
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
import os
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import random
import argparse

import sys
sys.path.append("..") 
from train.util import set_seed
from model import Gmm
from train.trainer import Trainer
from gmm import Gmm_np as GM

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,        
        #filename= 'dat.log',
)
logger = logging.getLogger(__name__)

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
        return temp #torch.from_numpy(self.data[idx, 3:].astype(np.float32))
    
    def info(self, idx=None):        
        return self.data[:,:3] if idx is None else  self.data[idx,:3]       
    
    def getitem(self, code, day):
        pos = np.where((self.data[:, 0]==code)&(self.data[:, 1]==str(day)))
        return self.__getitem__(pos[0][0])

def train(train_dataset, test_dataset, lastmodel=None):
    
    model_config = Gmm.get_default_config()
    model_config.ncluster = 512
    model_config.ndim = 10
    model_config.dataset = train_dataset
    model_config.initParaRange = 1024
    #model_config.dl_mode = False        
    model = Gmm(model_config)    
    
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-3 
    train_config.warmup_tokens = 0
    train_config.max_epochs = 40
    train_config.num_workers = 0
    train_config.batch_size = len(train_dataset)
    train_config.distributed = False
    #train_config.device = 'cpu'
    trainer = Trainer(train_config, model, train_dataset, test_dataset)
    if lastmodel is not None:
        trainer.load_lastpoint(lastmodel)
    trainer.train()     

def load_para(model_num):

    model_config = Gmm.get_default_config()
    model_config.ncluster = 512
    model_config.ndim = 10
    model_config.loadPara = True
    model = Gmm(model_config)

    train_config = Trainer.get_default_config()
    train_config.device = 'cpu'
    trainer = Trainer(train_config, model, None, None)    
    model = trainer.load_lastpoint(model_num)
    return model.param()
    
def test(dataset, model_num):
    model_config = Gmm.get_default_config()
    model_config.ncluster = 512
    model_config.ndim = 10
    model_config.loadPara = True
    model = Gmm(model_config)

    train_config = Trainer.get_default_config()
    #train_config.device = 'cpu'
    trainer = Trainer(train_config, model, None, None)    
    model = trainer.load_lastpoint(model_num)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                            batch_size=len(dataset), num_workers=0)
        
    data = next(iter(loader))
        
    w, _ = model(data)
    va, idx = w.max(dim=1)
    rv, ri = va.sort(descending=True)    
    m = dataset.info(ri[idx[ri]==418])
    print(m[:30], m.shape)
    
if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 1)')    
    args = parser.parse_args()    
    set_seed(55)
    
    if args.m == 0:    
        tr_path = '../data/gmm/train.npy'
        train_dataset = StockDataset(tr_path)        
        train(train_dataset, None, lastmodel=None)
        
    elif args.m == 1:
        te_path = '../data/gmm/test.npy'
        test_dataset = StockDataset(te_path)
        tr_path = '../data/gmm/train.npy'
        train_dataset = StockDataset(tr_path)  
        test(train_dataset, 39)
        #mu, var, Pi = load_para(10)                
        #w = GM.update_W(test_dataset[0], mu, var, Pi) 