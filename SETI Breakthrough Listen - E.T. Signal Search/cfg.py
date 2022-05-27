class CFG:
    model_test=True
    frac=0.01#0.02 #1.0#
    pretrained=True #False#
    model= "tf_efficientnet_b0"#"resnet18"#"tf_efficientnet_b0"#"resnet18"
    train_bs=16
    valid_bs=8#12
    test_bs=8
    epoch=2
    fold=[0]
    seed= 2022
    lr=2e-4
    es=4
    device="cuda"

    # make fold
    make_fold=True
    
    #train, inference
    _train=True
    _predict=True

    #file path
    # path_train="e:/kaggle_data/SETI Breakthrough Listen - E.T. Signal Search/seti_train_labelling.csv"
    # path_test="e:/kaggle_data/SETI Breakthrough Listen - E.T. Signal Search/sample_submission.csv"
    # path_ref_model='e:/kaggle_data/timm'
    path_train="../input/seti-breakthrough-listen/train_labels.csv"
    path_test="../input/seti-breakthrough-listen/sample_submission.csv"
    path_ref_model='../input/timm-pytorch-image-models/pytorch-image-models-master'

def get_cfg():
    return CFG