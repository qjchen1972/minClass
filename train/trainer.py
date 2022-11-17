"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import math
import logging
from tqdm import tqdm
import numpy as np
import os
import torch
from torch.utils.data.dataloader import DataLoader
from train.util import set_seed, dict2str
from train.util import CfgNode as CN

logger = logging.getLogger(__name__)

class ScheduledOptim(object):
    
    def __init__(self, optimizer, init_lr, n_warmup_steps, 
                 final_steps, n_current_steps=0):
                 
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = init_lr
        self.final_steps = final_steps

    def step_and_update_lr(self):        
        self._optimizer.step()
        self._update_learning_rate()
    '''
    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)    
    '''
    
    def _update_learning_rate(self):
        
        self.n_current_steps += 1
        if self.n_warmup_steps == 0:
            self.lr = self.init_lr
        else:
            if self.n_current_steps < self.n_warmup_steps:
                lr_mult = float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
            else:
                progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_steps - self.n_warmup_steps))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            self.lr = self.init_lr * lr_mult
            
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self.lr

class Trainer:

    @staticmethod
    def get_default_config():
    
        C = CN()
        # dataloder parameters, num_workers is 0 in windows
        C.num_workers = 0
        C.shuffle = True
        # optimizer parameters        
        C.max_epochs = 20
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        #multi GPU
        C.distributed = False
        C.warmup_tokens = 0
        C.ckpt_path = './models'
        C.device = 'auto'
        C.rndseed = 42
        return C
    
    def __init__(self, config, model, train_dataset, test_dataset=None):
    
        if config.distributed:
            self.local_rank, self.rank, self.world_size = self.init_distributed_mode()
        else:
            self.rank = 0
            self.world_size = 1
                    
        set_seed(config.rndseed + self.rank)
         
        if config.device == 'auto':
            self.device = torch.cuda.current_device()\
                          if torch.cuda.is_available() else "cpu"
        else:
            self.device = 'cpu'
        
        if self.device == 'cpu':
            config.distributed = False
        
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        self.start_epoch = 0
        self.best_loss = float('inf')
       
       
        self.model = self.model.to(self.device)
        
        if config.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if train_dataset is not None else None
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                                test_dataset) if test_dataset is not None else None
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                         device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            if config.shuffle:
                self.train_sampler = torch.utils.data.RandomSampler(train_dataset)\
                                if train_dataset is not None else None
            else:
                self.train_sampler = torch.utils.data.SequentialSampler(train_dataset)\
                                if train_dataset is not None else None
                                
            self.test_sampler = torch.utils.data.SequentialSampler(test_dataset) \
                                if test_dataset is not None else None
            if self.device != 'cpu':
                self.model = torch.nn.DataParallel(self.model).to(self.device)

    #同步数据    
    def reduce_dict(self, input_dict, average=True):
        if self.world_size < 2:
            return input_dict
        with torch.no_grad():
            names = []
            values = []
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            torch.distributed.all_reduce(values)
            if average:                                               
                values /= self.world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict
    
    def init_distributed_mode(self):
    
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ["LOCAL_RANK"])
        #device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        #torch.cuda.set_device(rank % torch.cuda.device_count())
        # windows下只支持gloo, linux下使用nccl
        torch.distributed.init_process_group(backend="gloo",#backend="nccl", 
                                             init_method='env://',
                                             world_size=world_size, 
                                             rank=rank)
        torch.distributed.barrier()        
        return local_rank, rank, world_size
        
    
    def save_model(self, name='best.pt'):
        if self.rank != 0: return
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        filepath = os.path.join(self.config.ckpt_path, name)
        logger.info("saving %s", filepath)
        torch.save(raw_model.state_dict(), filepath)
        
    def save_lastpoint(self, epoch, best_loss):
        if self.rank != 0: return
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        filepath = os.path.join(self.config.ckpt_path, 'model_%d.pt' %(epoch))
        torch.save({'epoch':epoch, 'best_loss':best_loss, 
                    'state_dict':raw_model.state_dict()},
                   filepath)
                   
        
    def load_lastpoint(self, epoch):
        filepath = os.path.join(self.config.ckpt_path, 'model_%d.pt' %(epoch))
        state_dict = torch.load(filepath)
        
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict['state_dict'])
        else:        
             self.model.load_state_dict(state_dict['state_dict'])        
        self.start_epoch = state_dict['epoch'] + 1
        self.best_loss = state_dict['best_loss']
        return self.model.module if hasattr(self.model, "module") else self.model
    
    
    def train(self, collate_fn=None):
        
        assert self.train_dataset, 'train dataset is None'        
        model, config = self.model, self.config
        
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.config_optimizers(config)
        if optimizer is not None:
            init_tokens = self.start_epoch * (len(self.train_dataset) //\
                          config.batch_size)
            final_steps = config.max_epochs * (len(self.train_dataset) //\
                          config.batch_size)            
            optim_schedule = ScheduledOptim(optimizer, 
                                        init_lr=config.learning_rate, 
                                        n_warmup_steps=config.warmup_tokens, 
                                        final_steps=final_steps,
                                        n_current_steps=init_tokens)
        
        def run_epoch(split):
            is_train = split == 'train'
            
            #若使用fasterrcnn, 验证集计算时依旧使用train状态
            #model.train()            
            model.train(is_train)
            
            (data, sampler)  = (self.train_dataset, self.train_sampler)\
                               if is_train else (self.test_dataset, self.test_sampler)         
            
            loader = DataLoader(data, pin_memory=True, #shuffle=True,
                                collate_fn = collate_fn, #lambda x: tuple(zip(*x))
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                sampler=sampler)
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))\
                   if is_train else enumerate(loader)
            lossName = []
            
            for it, x in pbar:
                # forward the model
                with torch.set_grad_enabled(is_train and optimizer is not None):
                    _, loss_dict = model(x)                    
                    loss = loss_dict['total_loss']                    
                    if it == 0: lossName = loss_dict.keys()                    
                    loss_dict_reduced = self.reduce_dict(loss_dict)
                    losses.append([loss_dict_reduced[k].item()\
                                   for k in loss_dict_reduced.keys()])
                                   
                if is_train: 
                
                    if optimizer is  None:
                    
                        pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                                {loss.item():.5f}")
                    else:
                    
                        # backprop and update the parameters
                        #optim_schedule.zero_grad()
                        model.zero_grad(set_to_none=True)
                        loss.backward()
                        #torch.nn.utils.clip_grad_norm_(model.parameters(),\
                        #                               config.grad_norm_clip)
                        optim_schedule.step_and_update_lr()                    
                        # report progress
                        pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                                {loss.item():.5f}. lr {optim_schedule.lr:e}")
                            
                '''
                if  is_train and optimizer is None:
                    pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                            {loss.item():.5f}")
                elif is_train and optimizer is not None:                    
                    pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                            {loss.item():.5f}. lr {optim_schedule.lr:e}")
                '''            
            
            if is_train and optimizer is None:
                raw_model.update()
                
            mean_loss = np.mean(losses, axis=0)
            dict_loss = dict(zip(lossName, mean_loss))
            if self.rank == 0:
                if is_train: 
                    logger.info('train loss: %s' % dict2str(dict_loss, 3)) #(str(dict_loss)))
                else:
                    logger.info('test loss: %s' % dict2str(dict_loss, 3)) #(str(dict_loss)))  

            return dict_loss['total_loss']
            

        for epoch in range(self.start_epoch, config.max_epochs):
            
            if config.distributed:
                self.train_sampler.set_epoch(epoch)

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                if test_loss < self.best_loss:
                    self.save_model()
                    self.best_loss = test_loss                
            self.save_lastpoint(epoch, self.best_loss)

