from cfg import get_cfg
CFG=get_cfg()

import torch
import torch.nn as nn
import sys
sys.path.append(CFG.path_ref_model)
import timm

def loss_fn(outputs,targets):
    # return nn.BCEWithLogitsLoss()(outputs,targets)
    return nn.CrossEntropyLoss()(outputs,targets)

def get_model(classes=1,pretrained=True):
    classes=classes
    base_model=timm.create_model(CFG.model, pretrained=pretrained, num_classes=100, in_chans=1)
    return base_model 