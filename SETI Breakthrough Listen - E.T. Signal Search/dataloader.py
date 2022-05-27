import random
import os
import numpy as np
from PIL import Image

from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader

from cfg import get_cfg
CFG=get_cfg()

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
        self.labels = df['label'].values
        self.dir_names = df['path'].values
        self.transform = transform
        self.conf = conf
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        file_path = self.df.loc[idx,"path"]
#1 method
#         image = np.load(file_path)#(6,273,256)
#         image=np.vstack(img).transpose((1,0))#(1638,256)->(256,1638)
#         if self.transform is not None:
#             image = self.transform(image=image)['image']
#         image = torch.tensor(image,dtype=torch.float32)
#         label = torch.tensor([self.labels[idx]]).float()

#2 method
        image = np.load(file_path)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        img_pl = Image.fromarray(image).resize((384, 384), resample=Image.BICUBIC)
        image = np.array(img_pl)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = torch.from_numpy(image).unsqueeze(dim=0)
        label = torch.tensor(self.labels[idx],dtype=torch.long)
        
        return {
            "x":image, 
            "y":label
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
