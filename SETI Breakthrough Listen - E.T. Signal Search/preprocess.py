from cfg import get_cfg
CFG=get_cfg()

import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
import albumentations as A

from sklearn.metrics import roc_auc_score

import os

def convert_image_id_2_path(image_id: str, is_train: bool = True) -> str:
    folder = "train" if is_train else "test"
    return "e:/kaggle_data/SETI Breakthrough Listen - E.T. Signal Search/{}/{}/{}.npy".format(
        folder, image_id[0], image_id 
    )
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def make_label(x):
    return np.int16(x*100)

def get_data():
    train=pd.read_csv(CFG.path_train)
    train["path"]=train["id"].apply(lambda x: convert_image_id_2_path(x))
    # train["level"]=train.preds.apply(lambda x: sigmoid(x))
    # train["label"]=train.level.apply(lambda x:make_label(x))

    if CFG.make_fold:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
        train["fold"]=-1
        for i,(_,val_idx) in enumerate(skf.split(train.sample(frac=1.),train.target)):
            train.loc[val_idx,"fold"]=i 

    return train    

def get_data_test():
    test = pd.read_csv(CFG.path_test)
    test["path"]=test["id"].apply(lambda x: convert_image_id_2_path(x,False))
    return test