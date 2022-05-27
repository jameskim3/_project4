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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge

if __name__ == "__main__":
    #train data
    train = get_data()

    # TF-IDF
    vec=TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (2,5),max_features=46000)
    X=vec.fit_transform(train.comment_text)

    # Model
    model=Ridge(alpha=0.5)
    model.fit(X,train["target"])

    #submission
    test = get_data_test()
    print(test.head())
    X_test=vec.transform(test.comment_text)
    test["score"]=model.predict(X_test)
    test[["comment_id","score"]].to_csv("submission.csv",index=False)
    print(test.head())

    f,ax=plt.subplots(1,1,figsize=(15,5))
    sns.histplot(test.score)
    plt.show()