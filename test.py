from __future__ import print_function

import sys
import math
import os
import random
import argparse
import numpy as np
import cv2
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils import cfg_from_file,cfg,preproc,Timer

import logging
from module import ImagenetDataset
from module.functions import Detect
from module.model.ssd_lite import build_ssd_lite,trainable_param,configure_optimizer,configure_lr_scheduler,find_previous,resume_checkpoint

logger = logging.getLogger('global')

"""cd 
Parse input arguments
"""
parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
parser.add_argument('--cfg', default='./config/ssd_lite_mobilenetv2_test_vid.yml',
            help='optional config file', type=str)

args = parser.parse_args()
   



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
    #dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -2))
    train_dataset = ImagenetDataset(cfg.DATASET_DIR, cfg.CACHE_FILE, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -2),is_val=True)
    logger.info("build dataset done")
    train_loader= DataLoader(train_dataset, 
                                    cfg.TRAIN_BATCH_SIZE, 
                                    num_workers=0,
                                    collate_fn=detection_collate, 
                                    pin_memory=True)   
    return train_loader

def test(train_loader, model,detector,output_dir):
    dataset = train_loader.dataset
    num_images = len(dataset)
    num_classes = detector.num_classes
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

    _t = Timer()

    for i in iter(range((num_images))):
        img = dataset.pull_image(i)
        scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(),requires_grad=False)
        _t.tic()
        # forward
        out = model(images, phase='eval')
        
        # detect
        detections = detector.forward(out)
        time = _t.toc()
        # TODO: make it smart:
        for j in range(1, num_classes):
            cls_dets = list()
            for det in detections[0][j]:
                if det[0] > 0:
                    d = det.cpu().numpy()
                    score, box = d[0], d[1:]
                    box *= scale
                    cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),1)
                    box = np.append(box, score)
                     
                    cls_dets.append(box)
            if len(cls_dets) == 0:
                cls_dets = empty_array
            all_boxes[j][i] = np.array(cls_dets)
        cv2.imwrite('a.jpg',img)

        



            # log per iter
        log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
                prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
                time=time)
        sys.stdout.write(log)
        sys.stdout.flush()

        # write result to pkl
    # with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #     # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
    # print('Evaluating detections')
    # train_loader.dataset.evaluate_detections(all_boxes, output_dir)


    
    
  
    
    #test_model()

def main():

    if args.cfg is not None:
         cfg_from_file(args.cfg)

    #creat model
    model = build_ssd_lite(cfg).cuda().eval()
    detector = Detect(cfg.POST_PROCESS, model.priors)

    # build dataset loader
    train_loader = build_data_loader(cfg.DATASET)

    
   
    start_epoch=0
    previous = find_previous(cfg.EXP_DIR)
    if previous:
        start_epoch = previous[0][-1]
        #print(model.state_dict())
        model = resume_checkpoint(model,previous[1][-1],cfg.TRAIN.RESUME_SCOPE)
        
    else:
        print(cfg.RESUME_CHECKPOINT)
        model=resume_checkpoint(model,cfg.RESUME_CHECKPOINT,cfg.TRAIN.RESUME_SCOPE)  #RESUME_CHECKPOINT
    test(train_loader, model,detector,cfg.EXP_DIR)
  
if __name__ == '__main__':

    main()
