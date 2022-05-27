class CFG:
    model_test=True
    frac=0.005#0.02 #1.0#
    pretrained=True #False#
    model= "tf_efficientnet_b0"#"resnet18"#"tf_efficientnet_b0"#"resnet18"
    train_bs=4
    valid_bs=4#12
    test_bs=4
    epoch=5
    fold=[0]
    seed= 2022
    lr=2e-4
    es=4
    device="cuda"

    # make fold
    make_fold=True
    
    #train, inference
    _train=True#True
    _predict=True

    #file path
    path_train="E:/kaggle_data/Jigsaw Rate Severity of Toxic Comments/jigsaw-toxic-comment-classification-challen/train.csv"
    path_test="E:/kaggle_data/Jigsaw Rate Severity of Toxic Comments/sample_submission.csv"
    path_test2="E:/kaggle_data/Jigsaw Rate Severity of Toxic Comments/comments_to_score.csv"
    path_ref_model="bert-base-uncased"
    
    # path_train="../input/jigsaw-toxic-comment-classification-challenge/train.csv"
    # path_test="../input/jigsaw-toxic-severity-rating/sample_submission.csv"
    # path_test2="../input/jigsaw-toxic-severity-rating/sample_submission.csv"
    # path_ref_model="../input/bert-base-uncased"

    #customize
    max_sens=512


def get_cfg():
    return CFG