from cfg import get_cfg
CFG=get_cfg()

import numpy as np
from model import get_model,loss_fn
import torch
import torch.nn as nn
from dataloader import get_dataset_test,get_dataloader_test
from tqdm import tqdm

def predict(model,dataloader):
    preds=[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            ids = data["ids"].to(CFG.device,non_blocking=True)
            mask = data["mask"].to(CFG.device,non_blocking=True)
            outputs=model(ids,mask)
            outputs = outputs.squeeze(-1)
            preds.append(outputs.cpu().detach().numpy())
        preds=np.concatenate(preds)

    return preds

def loop_predict(test,fold=0,save_model=True):
    # Dataset
    test_loader=get_dataloader_test(test)

    all_preds=[]
    for pth in CFG.fold:
        model=get_model()
        model.to(CFG.device)
        model.eval()
        path=f"{CFG.model}/model_{CFG.model}_fold_{fold}.bin"
        model.load_state_dict(torch.load(path))
        preds=predict(model,test_loader)
        all_preds.append(preds)
    
    p=np.array(all_preds).T
    preds=np.average(p,axis=1)
    test["target"]=preds
    return test
