from cfg import get_cfg
CFG=get_cfg()

import os
import cv2
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold

def clean(data):

    # Clean some punctutations
    data = re.sub('\n', ' ', data)
    # Remove ip address
    data = re.sub(r'(([0-9]+\.){2,}[0-9]+)',' ', data)
    
    data = re.sub(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3', data)
    # Replace repeating characters more than 3 times to length of 3
    data = re.sub(r'([*!?\'])\1\1{2,}',r'\1\1\1', data)
    # patterns with repeating characters 
    data = re.sub(r'([a-zA-Z])\1{2,}\b',r'\1\1', data)
    data = re.sub(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1', data)

    # Add space around repeating characters
    data = re.sub(' +', ' ', data)
    
    # Ex) I didn ' t -> I didn't
    data = re.sub(" ' ", "'", data)
    
    return data

def get_data():
    train=pd.read_csv(CFG.path_train)
    train.severe_toxic=train.severe_toxic*2
    train["target"]=(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)).astype(int)
    if CFG.make_fold:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
        train["fold"]=-1
        for i,(_,val_idx) in enumerate(skf.split(train.sample(frac=1.),train.target)):
            train.loc[val_idx,"fold"]=i 
    # train.comment_text=train.comment_text.apply(lambda x: clean(x))

    # test
    print(f"target value:{train.target.value_counts()}")
    return train    

def get_data_test():
    comments_to_score = pd.read_csv(CFG.path_test2)
    comments_to_score.columns=["id","comment_text"]
    comments_to_score["target"]=-1
    return comments_to_score