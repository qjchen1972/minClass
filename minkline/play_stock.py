#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import torchvision
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
from torch.utils.data import Dataset
from torchvision import models
from torchvision.ops import misc
from PIL import Image
import json
import shutil
import time

from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append("..")
from model import Rmin
from train.trainer import Trainer    
    
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,        
        #filename= 'dat.log',
)

class StockDataset(Dataset):
    
    def __init__(self, path ):
        self.data = np.load(path, allow_pickle=True)     
                 
    
    def __getitem__(self, idx):        
        imgpath = '%s_%s.png' %(self.data[idx][0], self.data[idx][1])
        label = int(self.data[idx][2])        
        if label == 1:
            imgpath = os.path.join('../data/min/right', imgpath)
        else:
            imgpath = os.path.join('../data/min/wrong', imgpath)
        
        img = Image.open(imgpath).convert('RGB')                 
        img = torchvision.transforms.ToTensor()(img)
        return img, torch.tensor(label, dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]        
            
    def info(self, idx):
        return self.data[idx]
        
        

def train(train_dataset, test_dataset, lastmodel=None):
    
    model_config = Rmin.get_default_config()
    model_config.model_name = 'resnet50'
    model = Rmin(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-5 
    train_config.warmup_tokens = 0
    train_config.max_epochs = 50
    train_config.num_workers = 0
    train_config.batch_size = 32
    train_config.distributed = False
    train_config.device = 'cpu'
    
    trainer = Trainer(train_config, model, train_dataset, test_dataset)
    
    if lastmodel is not None:
        trainer.load_lastpoint(lastmodel)    
    trainer.train()
    
def test(dataset):

    model_config = Rmin.get_default_config()
    model_config.model_name = 'resnet50'
    model = Rmin(model_config)

    '''
    train_config = Trainer.get_default_config()
    train_config.batch_size = 4
    train_config.distributed = False
    #train_config.device = 'cpu'
    trainer = Trainer(train_config, model, None, None)
    model = trainer.load_lastpoint(10)
    '''
    
    checkpoint = torch.load('./models/best.pt')
    model.load_state_dict(checkpoint)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, pin_memory=True, shuffle=False,
                        batch_size=1,
                        num_workers=0)
    temp = []
    for i,(img, target) in enumerate(loader):
        ans,_ = model(img)
        temp.append(dict(code=dataset.info(i)[0],
                         day=dataset.info(i)[1],
                         ret=int(dataset.info(i)[2]),
                         prd=ans.cpu().item()))
                        
    
    temp = pd.DataFrame(temp)
    print(temp)
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    t1 = temp[temp.prd>0.45]
    t2 = t1[t1.ret==1]
    print(t1,  t2)
    print(t2.shape[0] / t1.shape[0], t1.shape, t2.shape)   
    
    
#python -m torch.distributed.run --nproc_per_node=1  --nnodes=1 --node_rank=0 --master_addr="127.0.0.1"  --master_port=29500 play_stock.py
#python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1 --node_rank=0 play_stock.py
#python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 play_stock.py

if __name__ == '__main__':
    train_dir = '../data/min/train.npy'
    valid_dir = '../data/min/test.npy'
    train_dataset = StockDataset(train_dir)
    test_dataset = StockDataset(valid_dir)
    
    #train(train_dataset, test_dataset, 0)
    test(test_dataset)
    
    