import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from module.model.mobilenet import mobilenet_v2,mobilenet_v2_075, mobilenet_v2_050,mobilenet_v2_025
from module.functions import PriorBox
from module.loss import L2Norm,MultiBoxLoss


import os




class SSDLite(nn.Module):
    """Single Shot Multibox Architecture for embeded system
    See: https://arxiv.org/pdf/1512.02325.pdf & 
    https://arxiv.org/pdf/1801.04381.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, cfg,base, extras, head, feature_layer, priorbox,num_classes):
        super(SSDLite, self).__init__()
        self.priorbox = priorbox
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]
        self.criterion = MultiBoxLoss(cfg, self.priors)
        

    def forward(self, x, targets=None,phase='train'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources = list()
        loc = list()
        conf = list()

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            sources.append(x)
            # if k % 2 == 1:
            #     sources.append(x)

        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        print(phase)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        elif phase == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            loss_l, loss_c = self.criterion(output, targets)
            #print(loss_l.shape,loss_c.shape)
            return loss_l, loss_c
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

def add_extras(base, feature_layer, mbox, num_classes):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
        if layer == 'S':
            extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
            in_channels = depth
        elif layer == '':
            extra_layers += [ _conv_dw(in_channels, depth, stride=1, expand_ratio=1) ]
            in_channels = depth
        else:
            in_channels = depth
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, (loc_layers, conf_layers)

# based on the implementation in https://github.com/tensorflow/models/blob/master/research/object_detection/models/feature_map_generators.py#L213
# when the expand_ratio is 1, the implemetation is nearly same. Since the shape is always change, I do not add the shortcut as what mobilenetv2 did.
def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def build_ssd_lite(cfg):
    #base = cfg.MODEL.NETS
    base = mobilenet_v2
    feature_maps = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]  
    number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in cfg.MODEL.ASPECT_RATIOS]  

    base_, extras_, head_ = add_extras(base(), cfg.MODEL.FEATURE_LAYER,number_box, cfg.MODEL.NUM_CLASSES)

    priorbox = PriorBox(image_size=cfg.MODEL.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.MODEL.ASPECT_RATIOS, 
                    scale=cfg.MODEL.SIZES, archor_stride=cfg.MODEL.STEPS, clip=cfg.MODEL.CLIP)
    return SSDLite(cfg.MATCHER,base_, extras_, head_, cfg.MODEL.FEATURE_LAYER,  priorbox,cfg.MODEL.NUM_CLASSES)


def save_checkpoints(epochs, iters=None):
    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)
    if iters:
        filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
    else:
        filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
    filename = os.path.join(self.output_dir, filename)
    torch.save(model.state_dict(), filename)
    with open(os.path.join(output_dir, 'checkpoint_list.txt'), 'a') as f:
        f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
    print('Wrote snapshot to: {:s}'.format(filename))

# TODO: write relative cfg under the same page
def resume_checkpoint(model,resume_checkpoint, resume_scope):
    if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
        print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
        return False
    print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
    checkpoint = torch.load(resume_checkpoint)

    # print("=> Weigths in the checkpoints:")
    # print([k for k, v in list(checkpoint.items())])

    # remove the module in the parrallel model
    if 'module.' in list(checkpoint.items())[0][0]:
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        checkpoint = pretrained_dict

    
    #resume_scope = self.cfg.TRAIN.RESUME_SCOPE
    # extract the weights based on the resume scope
    if resume_scope != '':
        pretrained_dict = {}
        #print(list(checkpoint.items))
        for k, v in list(checkpoint.items()):
            #print(k,'***start***')
            for resume_key in resume_scope.split(','):
                if resume_key in k:
                    pretrained_dict[k] = v
                    #print(pretrained_dict[k])
                    break
                #     else:
                #         print(k)
                # print('***end***')
                  
        checkpoint = pretrained_dict

    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
    # print("=> Resume weigths:")
    # print([k for k, v in list(pretrained_dict.items())])

    checkpoint = model.state_dict()

    unresume_dict = set(checkpoint)-set(pretrained_dict)
    if len(unresume_dict) != 0:
        print("=> UNResume weigths:")
        print(unresume_dict)

    checkpoint.update(pretrained_dict)
    model.load_state_dict(checkpoint)

    return model


def find_previous(output_dir):
    if not os.path.exists(os.path.join(output_dir, 'checkpoint_list.txt')):
        return False
    with open(os.path.join(output_dir, 'checkpoint_list.txt'), 'r') as f:
        lineList = f.readlines()
    epoches, resume_checkpoints = [list() for _ in range(2)]
    for line in lineList:
        epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
        checkpoint = line[line.find(':') + 2:-1]
        epoches.append(epoch)
        resume_checkpoints.append(checkpoint)
    return epoches, resume_checkpoints

def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0



def trainable_param(model,trainable_scope):
    for param in model.parameters():
        param.requires_grad = False

    trainable_param = []
    for module in trainable_scope.split(','):
        if hasattr(model, module):
            # print(getattr(self.model, module))
            for param in getattr(model, module).parameters():
                param.requires_grad = True
            trainable_param.extend(getattr(model, module).parameters())

    return trainable_param


def configure_optimizer(trainable_param, cfg):
    if cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                        betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
    else:
        AssertionError('optimizer can not be recognized.')
    return optimizer



def configure_lr_scheduler(optimizer, cfg):
    if cfg.SCHEDULER == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'SGDR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
    else:
        AssertionError('scheduler can not be recognized.')
    return scheduler

