from cfg import get_cfg
CFG=get_cfg()

import os
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup

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
def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    
    soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
    only_text = soup.get_text()
    text = only_text
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string

    return text   

def get_data():
    test=pd.read_csv(CFG.path_train_test)
    test_label=pd.read_csv(CFG.path_train_test_label).replace(-1,0)
    test_merge=pd.merge(test,test_label,how="left",on="id")
    train=pd.read_csv(CFG.path_train)
    train=pd.concat([train,test_merge])

    cats = {'obscene': 0.16, 'toxic': 0.32, 'threat': 1.5, 
            'insult': 0.64, 'severe_toxic': 1.5, 'identity_hate': 1.5}
    for cat in cats:
        train[cat] = train[cat] * cats[cat]
    train["target"]=(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1))
    print(f"train_target_values:{train.target.value_counts()}")

    min_len=train[train.target>0].shape[0]
    print(f"min_len:{min_len}")
    under_sample=train[train.target==0].sample(n=min_len,random_state=CFG.seed)
    train=pd.concat([train[train.target>0],under_sample]).reset_index()
    print(f"train_new shape:{train.shape}")
    print(f"train_new_values:{train.target.value_counts()}")

    train["cat"]=(train.target*100).astype(int)
    if CFG.make_fold:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
        train["fold"]=-1
        for i,(_,val_idx) in enumerate(skf.split(train,train.cat)):
            train.loc[val_idx,"fold"]=i 
    
    print(f"fold 0:{train[train.fold==0].target.value_counts()}")
    print(f"fold 2:{train[train.fold==2].target.value_counts()}")
    print(f"fold 4:{train[train.fold==4].target.value_counts()}")

    if CFG.make_clean:
        train.comment_text=train.comment_text.apply(lambda x: text_cleaning(x))

    return train   
    
def get_data_test():
    comments_to_score = pd.read_csv(CFG.path_test2)
    comments_to_score.columns=["comment_id","comment_text"]
    comments_to_score["target"]=-1
    if CFG.make_clean:
        comments_to_score.comment_text=comments_to_score.comment_text.apply(lambda x: text_cleaning(x))
    return comments_to_score