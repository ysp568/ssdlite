from module.dataset import voc
from module.dataset import coco
from module.dataset import vid

dataset_map = {
                'voc': voc.VOCDetection,
                'coco': coco.COCODetection,
                'vid': vid.ImagenetDataset,
                'seq':vid.VIDDataset
            }

def gen_dataset_fn(name):
    """Returns a dataset func.

    Args:
    name: The name of the dataset.

    Returns:
    func: dataset_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in dataset_map:
        raise ValueError('The dataset unknown %s' % name)
    func = dataset_map[name]
    return func


import torch
import numpy as np

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


def collate_fn_new(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    images=[] 
    targets = []
    for index, sample in enumerate(batch):
        img_seq = sample[0]
        img_seq=torch.stack(img_seq,dim=0)
        images.append(img_seq)
        targets.append(sample[1])

        # if index ==0: 
        #     for i in range(len(target_seq)):
        #         seq_dict[f"index_{i}"]=[target_seq[i]]
        # else:
        #     for i in range(len(target_seq)):
        #         seq_dict[f"index_{i}"].append(target_seq[i])   
    return (torch.stack(images,dim=1) , targets)


from utils.data_augment import preproc
import torch.utils.data as data

def load_data(cfg, phase):
    if phase == 'train':
        if cfg.DATASET=='vid':
            dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.CACHE_FILE, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
            data_loader = data.DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, num_workers=0,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
        elif cfg.DATASET=='seq':
            dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.CACHE_FILE, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB),False,cfg.TRAIN_BATCH_SIZE)
            data_loader = data.DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, num_workers=12,
                                  shuffle=True,collate_fn=collate_fn_new, pin_memory=True)
        else:
            dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.CACHE_FILE, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
            data_loader = data.DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, num_workers=0,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    if phase == 'eval':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'test':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -2))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=0,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'visualize':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, 1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    return data_loader
