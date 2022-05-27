from cfg import get_cfg
CFG=get_cfg()

import numpy as np
from model import get_model,loss_fn
import torch
import torch.nn as nn
from dataloader import get_dataset,get_dataloader
import os

## Engine
from tqdm import tqdm
class Engine:
    def __init__(self,model,optimizer,scheduler=None):
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.device=CFG.device

    def loss_fn(self,outputs,targets):
        return nn.BCEWithLogitsLoss()(outputs,targets)        

    def train(self,data_loader):
        self.model.train()
        final_loss=0
        for data in tqdm(data_loader):
            self.optimizer.zero_grad()
            ids = data["ids"].to(CFG.device,non_blocking=True)
            mask = data["mask"].to(CFG.device,non_blocking=True)
            outputs=self.model(ids,mask)
            outputs = outputs.squeeze(-1)
            targets = data["targets"].to(CFG.device,non_blocking=True)
            loss=self.loss_fn(outputs,targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss/len(data_loader)
    
    def validate(self,data_loader):
        self.model.eval()
        final_loss=0
        for data in tqdm(data_loader):
            with torch.no_grad():
                ids = data["ids"].to(CFG.device,non_blocking=True)
                mask = data["mask"].to(CFG.device,non_blocking=True)
                outputs=self.model(ids,mask)
                outputs = outputs.squeeze(-1)
                targets = data["targets"].to(CFG.device,non_blocking=True)
                loss=self.loss_fn(outputs,targets)
                final_loss += loss.item()
        return final_loss/len(data_loader)

def loop_train(train,fold=0,save_model=True):
    # Dataset
    train_loader,valid_loader=get_dataloader(train,fold)
    model=get_model()
    model.to(CFG.device)

    # model folder
    if os.path.isdir(CFG.model) == False:
        os.makedirs(CFG.model)

    # Model,Optimizer, scheduler, engine
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=1e-5,mode="min",verbose=True)
    engine=Engine(model,optimizer,scheduler)
    best_loss=np.inf
    early_stopping=10
    early_stopping_cnt=0

    for epoch in range(CFG.epoch):
        train_loss=engine.train(train_loader)
        valid_loss=engine.validate(valid_loader)
        scheduler.step(valid_loss)

        if valid_loss<best_loss:
            best_loss=valid_loss
            torch.save(model.state_dict(),f"{CFG.model}/model_{CFG.model}_fold_{fold}.bin")
            print(f"fold={fold}, epoch={epoch}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")    
            early_stopping_cnt=0
        else:
            early_stopping_cnt+=1
        if early_stopping_cnt>early_stopping:
            break

    print(f"fold={fold}, best val loss={best_loss}")        