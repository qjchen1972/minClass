# coding: utf-8
import pandas as pd
import numpy as np
import os
import math
from train.util import set_seed
    
def data_daysplit(dirpath, srcfile, day, file1, file2):
    ans = np.load(os.path.join(dirpath, srcfile), allow_pickle=True)
    train = []
    test = []
    for one in ans:
        if int(one[1]) > day:
            test.append(one)
        else:
            train.append(one)
    print(len(train), len(test))        
    np.save(os.path.join(dirpath, file1), train)
    np.save(os.path.join(dirpath, file2), test)    

def data_ratesplit(dirpath, srcfile, rate, file1, file2):    
    ans = np.load(os.path.join(dirpath, srcfile), allow_pickle=True)
    train = np.random.permutation(ans.shape[0])[:int(ans.shape[0]*rate)]
    test = list(set(range(ans.shape[0])) - set(train))    
    print(len(train), len(test))        
    np.save(os.path.join(dirpath, file1), ans[train])
    np.save(os.path.join(dirpath, file2), ans[test])
    
    

if __name__ == '__main__':
    set_seed(42)
    
    #data_ratesplit('./data/kmean', 'type.npy', 0.8, 'train.npy', 'test.npy')
    #data_daysplit('./data/kmean', 'type.npy', 20220601, 'train.npy', 'test.npy')
    
    #minkline.npy
    #data_daysplit('./data/min', 'minkline.npy', 20220601, 'train.npy', 'test.npy')
    #ans = np.load(os.path.join('./data/min', 'train.npy'), allow_pickle=True)
    #print(ans, ans.shape)
    
    #MC
    #data_daysplit(dirpath='./data/MC', srcfile='type.npy', day=20220601, 
    #              file1='train.npy', file2='temp.npy')
    #data_daysplit(dirpath='./data/MC', srcfile='temp.npy', day=20220801, 
    #              file1='valid.npy', file2='test.npy')
    
    #GMM
    #data_ratesplit('./data/gmm', 'type.npy', 0.8, 'train.npy', 'test.npy')
    data_daysplit('./data/gmm', 'type.npy', 20220601, 'train.npy', 'test.npy')
        