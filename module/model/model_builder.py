# ssds part
from .ssd_lite import build_ssd_lite

# nets part

import torch

def _forward_features_size(model, img_size,batch_size):
    model.eval()
    x = torch.rand(batch_size, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, requires_grad=False) #.cuda()
    feature_maps = model(x, phase='feature')
    return [(o.size()[2], o.size()[3]) for o in feature_maps]


def create_model(cfg):
    '''
    '''
    base = cfg.NETS
    number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in cfg.ASPECT_RATIOS]  
        
    model = build_ssd_lite(base=base, feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES)
    #
    feature_maps = _forward_features_size(model, cfg.IMAGE_SIZE)
    print('==>Feature map size:')
    print(feature_maps)
    # 
    priorbox = PriorBox(image_size=cfg.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.ASPECT_RATIOS, 
                    scale=cfg.SIZES, archor_stride=cfg.STEPS, clip=cfg.CLIP)
    # priors = Variable(priorbox.forward(), volatile=True)

    return model, priorbox


