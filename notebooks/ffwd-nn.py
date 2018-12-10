#!/usr/bin/env python

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
## import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from copy import copy
import json

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, LeakyReLU, AlphaDropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras import regularizers

import tensorflow as tf

import keras.backend as K

from joblib import Parallel, delayed

def train_fold(ifold,val_idx,trn_data,val_data,**param):

    if "gpu_frac" in param:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=param["gpu_frac"])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
        
    num_round = 10000
    X_trn, y_trn = trn_data
    X_val, y_val = val_data


    L = Input(X_trn.shape[1:])
    inp = L 
    ## L = BatchNormalization()(L)
    for ilayer, layer_size in enumerate(param["layers"]):
        ## L = Dense(layer_size,use_bias=True,kernel_initializer="lecun_normal")(L)
        kernel_reg = None
        use_bias = True
        if ilayer == 1:
            kernel_reg = regularizers.l1(0.005)
            use_bias = False
        L = Dense(layer_size,use_bias=use_bias,kernel_regularizer=kernel_reg)(L)
        ## L = BatchNormalization()(L)
        if ilayer>1:
            L = Dropout(param["dropout"])(L)
            ## L = AlphaDropout(rate=param["dropout"])(L)
        L = LeakyReLU(alpha=0.2)(L)
        # # #L = Activation("selu")(L)
        
    out = Dense(1,use_bias=True)(L)    
    optimizer = Adam(lr=param["lr"],decay=param["lr_decay"])
    model = Model(inputs=inp,outputs=out)

    print(model.summary())
    model.compile(loss="mse",optimizer=optimizer,metrics=["mse"])
    y_prelim = model.predict(X_trn)
    print("isnan ---: ", np.isnan(y_prelim).sum())
    nans = X_trn[np.isnan(y_prelim).ravel()]
    for row in nans:
        print(row)
    np.save( "nan.npy", X_trn[np.isnan(y_prelim).ravel()] )

    ## assert(0)
    callbacks = [ EarlyStopping(patience=40,min_delta=0.002,verbose=True),
                  ## ReduceLROnPlateau(factor=0.2, patience=5,min_delta=0.005,vebose=True),
                  ModelCheckpoint('model-fold%d.hd5' % ifold, save_best_only=True),
                  ## TensorBoard(log_dir="./tensorboard_logs",histogram_freq=50,write_grads=True)
    ]
    model.fit( X_trn, y_trn, epochs=param["epochs"], batch_size=param["batch_size"], validation_data=(X_val,y_val), callbacks=callbacks )
    return val_idx, 'model-fold%d.hd5' % ifold
    
    

def train_function(**kwargs):

    global predictions
    global oof
    oof[:] = 0.
    do_predict = kwargs.get("do_predict",False)
    if do_predict:
        predictions[:] = 0.
    param = copy(default_param)
    param.update(kwargs)

    param["width"] = np.exp(param["width"])
    int_params = ["width","width_drop_start","width_drop_lenght","depth"]
    for par in int_params:
        param[par] = int(param[par])

    ## 128*2**
    width = param.pop("width")
    width_drop_start = param.pop("width_drop_start")
    width_drop_stop = width_drop_start + param.pop("width_drop_lenght")
    layers = []
    for ilayer in range(param.pop("depth")):
        layers.append(width)
        if ilayer >= width_drop_start and ilayer < width_drop_stop:
            width = max(8,width // 2)
    param["layers"] = layers
    param["lr"] = 10.**param["lr"]
    
    print(param)
    
    folds = KFold(n_splits=5, shuffle=True, random_state=15)
    start = time.time()
    feature_importance_df = pd.DataFrame()
    
    def get_data(trn_idx, val_idx):
        X_trn = train.iloc[trn_idx][features].values
        X_trn[ np.isnan(X_trn) | np.isinf(X_trn.astype(np.float)) ] = 0.
        y_trn = target.iloc[trn_idx]
        X_val = train.iloc[val_idx][features].values
        X_val[ np.isnan(X_val) | np.isinf(X_val.astype(np.float)) ] = 0.
        y_val = target.iloc[val_idx]

        return [(X_trn,y_trn),(X_val,y_val)]

    ## results = Parallel(n_jobs=2,verbose=10)(delayed(train_fold)(ifold,val_idx,*get_data(trn_idx, val_idx),**param)
    ##                                         for ifold, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)) )

    results  = [ train_fold(ifold,val_idx,*get_data(trn_idx, val_idx),**param) for ifold, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)) ]
    ## results  = [ (val_idx, "model-fold%d.hd5" % ifold) for ifold, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values))  ]

    
    X_train = train[features]
    for val_idx, model_name in results:        
        model = load_model(model_name)
        if do_predict:
            predictions += model.predict(X_test).ravel() / folds.n_splits
        X_val = train.iloc[val_idx][features].values
        X_val[ np.isnan(X_val) | np.isinf(X_val.astype(np.float)) ] = 0.
        oof[val_idx] = model.predict(X_val).ravel()
        
    ##   print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
    return -mean_squared_error(oof, target)**0.5
 
 

train = pd.read_hdf("../input/train_elo_world_nacat.hd5","train")
test = pd.read_hdf("../input/train_elo_world_nacat.hd5","test")

## print(train.isna().sum(),test.isna().sum())

target = pd.read_csv('../input/train.csv.zip',usecols=['target'],squeeze=True)
mean = target.mean()
std  = target.std() 
target -= mean
target /= std
print("target ---: ", target.mean(), target.std() )
predictions = np.zeros(test.shape[0])
oof = np.zeros(train.shape[0])

features  = [c for c in train.columns if c not in ['card_id', 'first_active_month'] and not "feature_" in c]
categorical_features = [c for c in features if 'feature_' in c]

mean_features = train[features].mean()
std_features = train[features].std()

train[features] -= mean_features
train[features] /= std_features

test[features] -= mean_features
test[features] /= std_features

from sklearn.preprocessing import  OneHotEncoder
for feat in categorical_features:
    enc = OneHotEncoder()
    enc.fit(train[[feat]])
    train_enc = pd.DataFrame(enc.transform(train[[feat]]))
    train_enc.columns = [feat+"_%d" % icat for icat in range(len(train_enc.columns)) ]
    test_enc = pd.DataFrame(enc.transform(test[[feat]]))
    test_enc.columns = train_enc.columns
    features += train_enc.columns.tolist()
    train = train.join(train_enc)
    test = test.join(test_enc)


default_param = {'dropout': 0.4,
                 'width': 5.6,
                 'depth': 6,
                 'width_drop_start': 1,
                 'width_drop_lenght': 3,
                 'batch_size': 4096,
                 ## 'lr':-2.5,
                 ## 'lr_decay':0.025,
                 'lr':-2.5,
                 'lr_decay':0.025,
                 'epochs':1000
                 ## 'gpu_frac':0.5,
}

X_test = test[features].values
X_test[ np.isnan(X_test) | np.isinf(X_test.astype(np.float)) ] = 0.


baseline = train_function(do_predict=True)
print("baseline: ---", baseline)


sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = predictions*std + mean
sub_df.to_csv("submit-nn-baseline.csv", index=False)

valid_df = pd.DataFrame({"card_id":train["card_id"].values})
valid_df["target"] = oof
valid_df.to_csv("valid-nn-baseline.csv", index=False)
### 
### 
### optimizer = BayesianOptimization(
###     f=train_function,                                                                                                                                                                     
###     pbounds={'dropout': (0.2,0.8),
###              'depth': (3,9),
###              'width': (np.log(100),np.log(500)),
###              'width_drop_start': (2,6),
###              'width_drop_lenght': (1,4),
###              'lr':(-4.5,-2.5),
###              'lr_decay':(0.015,0.05),
###     },                                                                                                                                                                                   
###     verbose=2,
###     random_state=23456,
### )
### 
### 
### if os.path.exists("./logs_nn_continue.json"):
###     with open("./logs_nn_continue.json") as fin:    
###         for point_str in fin.readlines():
###             point = json.loads(point_str)
###             optimizer.register(point["params"],point["target"])
### 
### logger = JSONLogger(path="./logs_nn.json")
### optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
### 
### 
### print("before starting maximum --- ", optimizer.max)
### 
### optimizer.maximize(init_points=10,
###                    n_iter=50,kappa=5.)
### 
### 
### print("after search maximum --- ", optimizer.max)
### 
### 
### train_function(**(optimizer.max["params"]))
### sub_df = pd.DataFrame({"card_id":test["card_id"].values})
### sub_df["target"] = predictions*std + mean
### sub_df.to_csv("submit-nn-opt.csv", index=False)
### 
### valid_df = pd.DataFrame({"card_id":train["card_id"].values})
### valid_df["target"] = oof
### valid_df.to_csv("valid-nn-opt.csv", index=False)

