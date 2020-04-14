from __future__ import print_function

import sys
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
from utils import cfg_from_file,cfg
from distributed import dist_init, DistModule, reduce_gradients,average_reduce, get_rank, get_world_size
import logging
from module import ImagenetDataset
from module.model.ssd_lite import build_ssd_lite

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

def build_data_loader(cfg):
    logger.info("build train dataset")
    # train_dataset
    train_dataset = ImagenetDataset(cfg.DATASET_DIR, cfg.CACHE_FILE, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
    logger.info("build dataset done")
    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)



    train_loader= data.DataLoader(train_dataset, 
                                    cfg.TRAIN_BATCH_SIZE, 
                                    num_workers=0,
                                    shuffle=True, 
                                    collate_fn=detection_collate, 
                                    pin_memory=True,
                                    sampler=train_sampler)
    

   
    return train_loader

#def train():
    
    
    #test_model()

def main():
    print('done')
    
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
    model = build_ssd_lite(cfg).cuda().train()
    print(model.priors.device)
    

if __name__ == '__main__':
    seed_torch(seed=123456)
    
    main()
