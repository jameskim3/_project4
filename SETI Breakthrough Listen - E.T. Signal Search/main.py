from cfg import get_cfg
from predict import loop_predict
CFG=get_cfg()

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from model import get_model,loss_fn
from preprocess import get_data,get_data_test
from dataloader import get_dataset,get_dataloader
from train import loop_train
from predict import loop_predict


if __name__ == "__main__":
    # #train data
    train = get_data()
    print(train.head())

    if CFG._train:
        for fold in CFG.fold:
            print(f"fold :{fold}")
            loop_train(train=train,fold=fold,save_model=True)
    
    if CFG._predict:
        test = get_data_test()
        result = loop_predict(test=test)
        result[["id","target"]].to_csv("submission.csv",index=False)
        result.head()
        
    # #show graph
    # f,ax=plt.subplots(1,2,figsize=(15,5))
    # sns.distplot(train.level,ax=ax[0])
    # sns.distplot(train.label,ax=ax[1])
    # plt.show()
    # f,ax=plt.subplots(5,1,figsize=(20,10))
    # for i in range(5):
    #     img=np.load(train.loc[i,"path"])#(6,273,256)
    #     img=img.astype(np.float32)
    #     img=np.vstack(img).transpose((1,0))#(1638,256)->(256,1638)
    #     ax[i].imshow(img)
    # plt.show()

    # #dataset
    # train_dataset,valid_dataset=get_dataset(train,0)
    # print(train_dataset[0])

    # #dataloader
    # train_loader,valid_loader=get_dataloader(train,0)
    # for a in train_loader:
    #     print(a["x"].shape, a["y"].shape)
    #     break    

    # #model
    # device="cuda"
    # model=get_model()
    # model.to(device)
    # model.train()
    # for a in train_loader:
    #     x=a["x"].to(device)
    #     y=a["y"].to(device)
    #     output=model(x)
    #     loss=loss_fn(output,y)
    #     break
    # print(output.squeeze(-1).shape)



