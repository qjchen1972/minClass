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

from model import Kmodel
from train.trainer import Trainer
from train.util import set_seed
from normal import Normal
from kclass import Kclass

os.environ['NUMEXPR_MAX_THREADS'] = '16'

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

    model_config = Kmodel.get_default_config()
    model_config.ncluster = 1024
    model_config.ndim = 10
    model_config.dataset = train_dataset
    model_config.dl_mode = False
    model_config.initParaRange = 102400 #len(train_dataset)
    model_config.dead_cluster = 0
    model_config.loadPara = False if lastmodel is None else True
    model_config.useVar = False
    model = Kmodel(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-3 
    train_config.warmup_tokens = 0
    train_config.max_epochs = 140
    train_config.num_workers = 0
    train_config.shuffle = False
    #train_config.rndseed = 7612
    train_config.batch_size = 40960#len(train_dataset)
    train_config.distributed = False
    #train_config.device = 'cpu'
    trainer = Trainer(train_config, model, train_dataset, test_dataset)
    if lastmodel is not None:
        trainer.load_lastpoint(lastmodel)
    trainer.train()            

    
def load_para(model_num, useVar):

    model_config = Kmodel.get_default_config()
    model_config.ncluster = 1024
    model_config.ndim = 10
    model_config.loadPara = True
    model_config.useVar = useVar
    model = Kmodel(model_config)

    train_config = Trainer.get_default_config()
    train_config.device = 'cpu'
    trainer = Trainer(train_config, model, None, None)
    
    model = trainer.load_lastpoint(model_num)
    return model.param()


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 0)')    
    args = parser.parse_args()    
    
    set_seed(55)
    tr_path = 'e:/img/train.npy'
    te_path = 'e:/img/test.npy'
    
    if args.m == 0:        
        train_dataset = StockDataset(tr_path)
        test_dataset = StockDataset(te_path)
        train(train_dataset, test_dataset, lastmodel=None)
        
    elif args.m == 1:
        mu, var = load_para(139, False)
        
        tr_config = Kclass.get_default_config()
        tr_config.mu = mu
        tr_config.var = var    
        tr_config.anspath = '../data/tr.pkl'        
        trcl = Kclass(tr_config)
        trcl.create_class(train_dataset)
        
        
        te_config = Kclass.get_default_config()
        te_config.mu = mu
        te_config.var = var   
        te_config.anspath = '../data/te.pkl'        
        tecl = Kclass(te_config)
        test_dataset = StockDataset(te_path)                  
        tecl.create_class(test_dataset)        
        
    elif args.m == 2:
        
        data = train_dataset.getitem('000100', 20220512)
        
        tr_config = Kclass.get_default_config()
        tr_config.mu = mu
        tr_config.var = var    
        tr_config.anspath = '../data/tr.pkl'        
        trcl = Kclass(tr_config)        
        ans = trcl.findwithdata(data)
        print(ans)
        
        te_config = Kclass.get_default_config()
        te_config.mu = mu
        te_config.var = var   
        te_config.anspath = '../data/te.pkl'        
        tecl = Kclass(te_config)
        ans = tecl.findwithdata(data)
        print(ans)        
    else:
        pass
        
