# -*- coding:utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import os
import math
import torch
from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import  MultivariateNormal

import sys
sys.path.append("..")
from train.util import torch2np

import logging
logger = logging.getLogger(__name__)


class Gmm_torch:

    diag = False
    
    def __init__(self):
        pass
    
    @classmethod
    def update_W(cls, X, Mu, Var, Pi):
        n_points, n_clusters = X.shape[0], Pi.shape[0]
        pdfs = torch.zeros(((n_points, n_clusters))).to(X)        
        for i in range(n_clusters):  
            try:
                if cls.diag:
                    var = torch.clamp(torch.diag(Var[i]), min=torch.finfo(torch.float32).eps)
                    Var[i] = torch.diag_embed(var)
                normal = MultivariateNormal(Mu[i], Var[i])
                pdfs[:, i] = Pi[i] * normal.log_prob(X).exp()  
            except ValueError as e:
                pdfs[:, i] = torch.finfo(torch.float32).eps
                logger.info('iter %d, multivariate_normal.pdf err' % (i))
               
        fact = pdfs.sum(dim=-1).clamp(min=torch.finfo(torch.float32).eps)        
        W = pdfs / fact[:, None]       
        return W
    
        
    @classmethod
    def update_Pi(cls, W):
        Pi = W.sum(dim=0) / W.sum()
        return Pi
    
    @classmethod
    def update_Mu(cls, X, W):
        n_clusters = W.shape[1]
        Mu = torch.zeros((n_clusters, X.shape[1])).to(X)
        for i in range(n_clusters):            
            Mu[i] = torch.mv(X.T, W[:, i]) / W[:, i].sum()
        return Mu
        
    @classmethod
    def update_Var(cls, X, W):
        n_clusters = W.shape[1]        
        Var = []
        for i in range(n_clusters):
            v = torch.cov(X.T, aweights=W[:, i]) 
            Var.append(v)
        return torch.stack(Var, dim=0)  
        
    #仿numpy的cov函数   
    @classmethod
    def update_Var1(cls, X, W):
        n_clusters = W.shape[1]        
        Var = []
        for i in range(n_clusters):
            avg = torch.mv(X.T, W[:, i]) / W[:, i].sum()
            X = X - avg
            m = X.T * W[:, i]
            fact =  W[:, i].sum() -  (W[:, i] * W[:, i]).sum() / W[:, i].sum()
            v = torch.mm(m, X)  / fact
            Var.append(v)
        return torch.stack(Var, dim=0)   
    
    @classmethod    
    def logLH(cls, X, Pi, Mu, Var):
        n_points, n_clusters = X.shape[0], Pi.shape[0]
        pdfs = torch.zeros(((n_points, n_clusters)))
        for i in range(n_clusters):
            try:
                normal = MultivariateNormal(Mu[i], Var[i])
                #normal = MultivariateNormal(Mu[i], scale_tril=torch.diag(Var[i]))
                pdfs[:, i] = Pi[i] * normal.log_prob(X).exp()   
            except ValueError as e:
                pdfs[:, i] = torch.finfo(torch.float32).eps
        return torch.log(pdfs.sum(dim=1)).mean()
    
    
class Gmm_np:
        
    def __init__(self):
        pass
        
    @classmethod
    def update_W(cls, X, Mu, Var, Pi):
    
        X = torch2np(X)
        Mu = torch2np(Mu)
        Var = torch2np(Var)
        Pi = torch2np(Pi)
        
        n_points, n_clusters = X.shape[0], Pi.shape[0]
        pdfs = np.zeros(((n_points, n_clusters)))        
        for i in range(n_clusters):  
            try:
                pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i],
                                   allow_singular=True)                                                   
            except ValueError as e:
                pdfs[:, i] = np.finfo(np.float32).eps  
                logger.info('iter %d, multivariate_normal.pdf err' % (i)) 
        fact = np.maximum(pdfs.sum(axis=1), np.finfo(np.float32).eps)       
        W = pdfs / fact[:, None]
        return W

    @classmethod
    def update_Pi(cls, W):
        Pi = W.sum(axis=0) / W.sum()
        return Pi
    
    @classmethod
    def update_Mu(cls, X, W):
        X = torch2np(X)
        n_clusters = W.shape[1]
        Mu = [] 
        for i in range(n_clusters):
            Mu.append(np.average(X, axis=0, weights=W[:, i]))  
        return np.stack(Mu, axis=0)
        
    @classmethod
    def update_Var(cls, X, W):
        X = torch2np(X)
        n_clusters = W.shape[1]        
        Var = []
        for i in range(n_clusters):
            Var.append(np.cov(X, rowvar=0, aweights=W[:, i]))
        return np.array(Var)   
    
    @classmethod    
    def logLH(cls, X, Pi, Mu, Var):
    
        X = torch2np(X)
        n_points, n_clusters = X.shape[0], Pi.shape[0]
        pdfs = np.zeros(((n_points, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i],
                                    allow_singular=True)
        return np.mean(np.log(pdfs.sum(axis=1)))
    

if __name__ == '__main__':    
    torch.set_printoptions(threshold=np.inf)    
    np.set_printoptions(precision=3, threshold=np.inf)
    pass
    
    
    