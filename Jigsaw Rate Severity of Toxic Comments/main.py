from cfg import get_cfg
from predict import loop_predict
CFG=get_cfg()

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from model import get_model,loss_fn
from preprocess import get_data,get_data_test,clean
from dataloader import get_dataset,get_dataloader
from train import loop_train
from predict import loop_predict

import transformers

if __name__ == "__main__":

    #train data
    train = get_data()
    print(train.head(10),train.shape)

    if CFG._train:
        for fold in CFG.fold:
            print(f"fold :{fold}")
            loop_train(train=train,fold=fold,save_model=True)
    
    if CFG._predict:
        test = get_data_test()
        result = loop_predict(test=test)
        result.columns=["comment_id","text","score"]
        result[["comment_id","score"]].to_csv("submission.csv",index=False)
        result.head()
        
    



