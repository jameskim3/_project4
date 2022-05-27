import random
import os
import numpy as np
from PIL import Image

from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader

from cfg import get_cfg
CFG=get_cfg()

import transformers

def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
random_seed(CFG.seed)

class SetiDataset(Dataset):
    def __init__(self, df, transform=None, conf=None):
        self.df = df.reset_index(drop=True)
        self.labels = df['target'].values
        # self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.path_ref_model)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        sentence = self.df.loc[idx, 'comment_text']
        bert_sens = self.tokenizer.encode_plus(
                                sentence,
                                truncation=True,
                                add_special_tokens = True, 
                                max_length = CFG.max_sens,#2502, # 上で314に設定しています
                                padding='max_length')
        ids = torch.tensor(bert_sens['input_ids'], dtype=torch.long)
        mask = torch.tensor(bert_sens['attention_mask'], dtype=torch.long)
        target = torch.tensor(self.labels[idx],dtype=torch.float)
        
        return {
            'ids': ids,
            'mask': mask,
            'targets': target
        }

def get_dataset(train=None,fold=0):
    train_df=train[train.fold!=fold].reset_index(drop=True).sample(frac=CFG.frac)
    valid_df=train[train.fold==fold].reset_index(drop=True).sample(frac=CFG.frac)
    train_dataset=SetiDataset(train_df)
    valid_dataset=SetiDataset(valid_df)
    return train_dataset,valid_dataset

def get_dataloader(train=None,fold=0):
    train_dataset,valid_dataset = get_dataset(train,fold)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=CFG.train_bs,num_workers=0,shuffle=True)
    valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=CFG.valid_bs,num_workers=0,shuffle=False)
    return train_loader,valid_loader

def get_dataset_test(test=None,fold=0):
    test_dataset=SetiDataset(test.sample(frac=1.))
    return test_dataset

def get_dataloader_test(test=None,fold=0):
    test_dataset=get_dataset_test(test=test)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=CFG.test_bs,num_workers=0,shuffle=False)
    return test_loader
