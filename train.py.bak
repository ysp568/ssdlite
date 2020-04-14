from __future__ import print_function

import sys
import math
import os
import random
import argparse
import numpy as np
import cv2
from datetime import datetime
#os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils import cfg_from_file,cfg,preproc,Timer

from distributed import dist_init, DistModule, reduce_gradients,average_reduce, get_rank, get_world_size
import logging
from module import ImagenetDataset
from module.model.ssd_lite import build_ssd_lite,trainable_param,configure_optimizer,configure_lr_scheduler,find_previous,resume_checkpoint

logger = logging.getLogger('global')

"""cd 
Parse input arguments
"""
parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
parser.add_argument('--cfg', default='./config/ssd_lite_mobilenetv2_train_vid.yml',
            help='optional config file', type=str)
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()
   

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

def build_data_loader(cfg):
    logger.info("build train dataset")
    # train_dataset
    train_dataset = ImagenetDataset(cfg.DATASET_DIR, cfg.CACHE_FILE, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
    logger.info("build dataset done")
    train_sampler = None
    if get_world_size() > 1:
         train_sampler = DistributedSampler(train_dataset)

    train_loader= DataLoader(train_dataset, 
                                    cfg.TRAIN_BATCH_SIZE, 
                                    num_workers=0,
                                    collate_fn=detection_collate, 
                                    pin_memory=True,
                                    sampler=train_sampler)
    
    return train_loader

def train(train_loader, model, optimizer, lr_scheduler,start_epoch):
    #cur_lr = lr_scheduler.get_cur_lr()
    max_epochs = cfg.TRAIN.MAX_EPOCHS
    rank = get_rank()

    warm_up = cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    #print(' train done')
    # num_per_epoch = len(train_loader.dataset) // \
    #     cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
   
    


    # if not os.path.exists(cfg.EXP_DIR) and \
    #         get_rank() == 0:
    #     os.makedirs(cfg.EXP_DIR)
    for epoch in iter(range(start_epoch+1, max_epochs+1)):
        
        if epoch & epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
            print('save')
            #save_checkpoints(epoch)
        if epoch > warm_up:
            lr_scheduler.step(epoch-warm_up)
        epoch_size = len(train_loader)

        _t = Timer()
        loc_loss = 0
        conf_loss = 0

        for idx,(images,targets) in enumerate(train_loader):
            print(images.shape)
            #print(targets.shape)
            

            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), requires_grad=False) for anno in targets]

            #forward
            _t.tic()
            loss_l,loss_c = model(images,targets)
            loss = loss_l + loss_c
            print(loss_l.item(),loss_c.item(),loss.item())

            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                loss.backward()
                reduce_gradients(model)
                # clip gradient
                #clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()
            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            if rank==0:
                log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*idx/epoch_size)) + '-'*int(round(10*(1-idx/epoch_size))), iters=idx, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())
                sys.stdout.write(log)
                sys.stdout.flush()

        # log per epoch
        if rank == 0:
            sys.stdout.write('\r')
            sys.stdout.flush()
            lr = optimizer.param_groups[0]['lr']
            log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
            sys.stdout.write(log)
            sys.stdout.flush()
  
    
    #test_model()

def main():
    # print('done')
    
    rank, world_size = dist_init()
    
    print("init done rank{}/world_size{}".format(rank,world_size))
    
    if args.cfg is not None:
         cfg_from_file(args.cfg)

    if rank == 0:
        print(cfg)
        if not os.path.exists(cfg.EXP_DIR):
             os.makedirs(cfg.EXP_DIR)
    #creat model
    #model = ModelBuilder().cuda().train()
    #print(cfg)
    model = build_ssd_lite(cfg).cuda().train()
    #print(model.state_dict())
    #print(cfg.TRAIN.TRAINABLE_SCOPE)

    # build dataset loader
    train_loader = build_data_loader(cfg.DATASET)

    # trainable_param
    _param = trainable_param(model,cfg.TRAIN.TRAINABLE_SCOPE)
    optimizer = configure_optimizer(_param, cfg.TRAIN.OPTIMIZER)
    #trainable_param(model,trainable_scope)
    print(model.priors.device)

    exp_lr_scheduler = configure_lr_scheduler(optimizer, cfg.TRAIN.LR_SCHEDULER)
   
    start_epoch=0
    previous = find_previous(cfg.EXP_DIR)
    if previous:
        start_epoch = previous[0][-1]
        #print(model.state_dict())
        model = resume_checkpoint(model,previous[1][-1],cfg.TRAIN.RESUME_SCOPE)
        #print(model.state_dict())
    else:
        print(cfg.RESUME_CHECKPOINT)
        model=resume_checkpoint(model,cfg.RESUME_CHECKPOINT,cfg.TRAIN.RESUME_SCOPE)  #RESUME_CHECKPOINT
    #print(model.state_dict())
    #dist_model = model
    dist_model = DistModule(model)
    logger.info("model prepare done")
    train(train_loader, dist_model, optimizer, exp_lr_scheduler,start_epoch)
  
if __name__ == '__main__':
    seed_torch(seed=123456)  
    main()
