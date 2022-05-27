from cfg import get_cfg
CFG=get_cfg()

import torch
import torch.nn as nn
import sys
sys.path.append(CFG.path_ref_model)
import transformers

def loss_fn(outputs,targets):
    # return nn.BCEWithLogitsLoss()(outputs,targets)
    return nn.BCEWithLogitsLoss()(outputs,targets)

def get_model(classes=1,pretrained=True):
    model = transformers.BertForSequenceClassification.from_pretrained(CFG.path_ref_model,num_labels=1)
    return model 