from cfg import get_cfg
CFG=get_cfg()

import torch
import torch.nn as nn
import sys
sys.path.append(CFG.path_ref_model)
import transformers

class JigsawModel(nn.Module):
    def __init__(self, model_name):
        super(JigsawModel, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(CFG.path_ref_model)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, 1)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,
                         attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs

def loss_fn(outputs,targets):
    # return nn.BCEWithLogitsLoss()(outputs,targets)
    return nn.BCEWithLogitsLoss()(outputs,targets)

def get_model(classes=1,pretrained=True):
    model = JigsawModel(CFG.path_ref_model)
    return model 