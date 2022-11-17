import math
import logging
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torchvision.ops import misc

import sys
sys.path.append("..")
from train.util import CfgNode as CN
logger = logging.getLogger(__name__)

class Rmin(nn.Module):

    @staticmethod
    def get_default_config():
        C = CN()
        C.model_name = 'resnet50'
        C.class_num = 1
        return C
    
    def __init__(self, config):
        super().__init__()
                
        self.body = models.resnet.__dict__[config.model_name]( pretrained=True, 
                                                               norm_layer=misc.FrozenBatchNorm2d)
        kernelCount = self.body.fc.in_features		
        self.body.fc = nn.Sequential(nn.Linear(kernelCount, config.class_num), nn.Sigmoid())
        #self.apply(self._init_weights) 
        logger.info("number of parameters: %e( %e )", 
                    sum(p.numel() for p in self.parameters()),
                    sum(p.numel() for p in self.parameters() if p.requires_grad))
    
         
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)  
        '''                
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if m.bias is not None: m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        '''            
        
        
    def config_optimizers(self, train_config):

        lr = train_config.learning_rate
        optim_groups = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    optim_groups += [{'params': [value], 
                                      'weight_decay': 0}]
                else:
                    optim_groups += [{'params': [value], 
                                      'weight_decay': train_config.weight_decay}]
        
        optimizer = torch.optim.AdamW(optim_groups, 
                                      lr=train_config.learning_rate, 
                                      betas=train_config.betas)
        '''
        optimizer = torch.optim.Adam(optim_groups, 
                                     lr=train_config.learning_rate, 
                                     betas=train_config.betas)
        '''
        #optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate, momentum=0.9)
        
        return optimizer
    
    def dev(self):
        for p in self.parameters():                    
            return p.device
        return 'cpu'    
    
    
    def forward(self, x):
        dev = self.dev()   
        if isinstance(x, torch.Tensor):
            image = x
            target = None
        elif isinstance(x, list):
            if len(x) >= 2:
                image = x[0]
                target = x[1]
            else:
                image = x[0]
                target = None
        image = image.to(dev)
        ans = self.body(image)
        dict_loss = None
        if target is not None:
            target = target.to(dev)
            loss = F.binary_cross_entropy(ans, target[:, None]) 
            dict_loss = dict(total_loss=loss)            
        return ans, dict_loss
        
