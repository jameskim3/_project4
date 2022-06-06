
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class DatasetTabular:
    def __init__(self,dataset,features):
        self.dataset=dataset
        self.feature=features
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,item):
        return {
            "x":torch.tensor(self.dataset[item,:],dtype=torch.float),
            "y":torch.tensor(self.feature[item,:],dtype=torch.float)
        }


## Model
class ModelTabular(nn.Module):
    def __init__(self,num_features,num_targets,num_layers,hidden_size,dropout):
        super().__init__()
        layers=[]
        for _ in range(num_layers):
            if len(layers)==0:
                layers.append(nn.Linear(num_features,hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size,hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size,num_targets))

        self.model=nn.Sequential(*layers)

    def forward(self,x):
        x=self.model(x)
        return x

## Engine

class Engine:
    def __init__(self,model,optimizer,device):
        self.model=model
        self.optimizer=optimizer
        self.device=device   
    
    def loss_fn(self,targets,outputs):
        return nn.BCEWithLogitsLoss()(outputs,targets)
    
    def train(self,data_loader):
        self.model.train()
        final_loss=0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs=data["x"].to(self.device)
            targets=data["y"].to(self.device)
            outputs=self.model(inputs)
            loss=self.loss_fn(targets,outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss/len(data_loader)
    
    def validate(self,data_loader):
        self.model.eval()
        final_loss=0
        for data in data_loader:
            inputs=data["x"].to(self.device)
            targets=data["y"].to(self.device)
            outputs=self.model(inputs)
            loss=self.loss_fn(targets,outputs)
            final_loss += loss.item()
        return final_loss/len(data_loader)
    
    def predict(self,data_loader):
        self.model.eval()
        final_predictions = []
        for data in data_loader:
            inputs=data["x"].to(self.device)
            predictions = self.model(inputs)
            predictions = predictions.cpu()
            final_predictions.append(predictions.detach().numpy())
        return final_predictions


def train_fold(fold,save_model=False):
    # select cols
    targets_cols=df_scored.drop(["sig_id","kfold"],axis=1).columns
    features_cols=df_features.drop(["sig_id"],axis=1).columns

    # Data Merge
    df_all=df_features.merge(df_scored,on="sig_id",how="left")

    # Dataset
    train_df=df_all[df_all.kfold!=fold].reset_index(drop=True)
    valid_df=df_all[df_all.kfold==fold].reset_index(drop=True)

    x_train=train_df[features_cols].to_numpy()
    x_valid=valid_df[features_cols].to_numpy()
    y_train=train_df[targets_cols].to_numpy()
    y_valid=valid_df[targets_cols].to_numpy()

    # DataLoader
    train_dataset=MoaDataset(x_train,y_train)
    train_loader=torch.utils.data.DataLoader(
        train_dataset,batch_size=1024,num_workers=8,shuffle=True
    )
    valid_dataset=MoaDataset(x_valid,y_valid)
    valid_loader=torch.utils.data.DataLoader(
        valid_dataset,batch_size=1024,num_workers=8,shuffle=False
    )
    
    # Model,Optimizer, scheduler, engine
    model=Model(
        num_features=x_train.shape[1],
        num_targets=y_train.shape[1],
        num_layers=5,
        hidden_size=2048,
        dropout=0.3        
    )
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(device,f"inputs:{x_train.shape[1]}, targets:{y_train.shape[1]}")
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=1e-5,mode="min",verbose=True
    )

    engine=Engine(model,optimizer,device)
    best_loss=np.inf
    early_stopping=10
    early_stopping_cnt=0
    EPOCH=300
    for epoch in range(EPOCH):
        train_loss=engine.train(train_loader)
        valid_loss=engine.validate(valid_loader)
        scheduler.step(valid_loss)

        if valid_loss<best_loss:
            best_loss=valid_loss
            torch.save(model.state_dict(),f"model_fold_{fold}.bin")
            print(f"fold={fold}, epoch={epoch}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")    
            early_stopping_cnt=0
        else:
            early_stopping_cnt+=1
        if early_stopping_cnt>early_stopping:
            break

    print(f"fold={fold}, best val loss={best_loss}")

def predict_fold(fold):
    df=df_test#pd.read_csv("./test_dummies.csv")
    features_cols=df.drop(["sig_id"],axis=1).columns
    x_test=df[features_cols].to_numpy()
    y_test=np.zeros((df.shape[0],206))
    test_dataset=MoaDataset(x_test,y_test)
    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=1024,num_workers=8,shuffle=False
    )
    

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=Model(
        num_features=x_test.shape[1],
        num_targets=y_test.shape[1],
        num_layers=5,
        hidden_size=2048,
        dropout=0.3        
    )
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(device,f"inputs:{x_test.shape[1]}, targets:{y_test.shape[1]}")

    model_save_path=f"./model_fold_{fold}.bin"
    model.load_state_dict(torch.load(model_save_path))
    model=model.to(device)
    
    engine=Engine(model,None,device)
    preds=engine.predict(test_loader)
    preds=np.vstack(preds)
    return preds