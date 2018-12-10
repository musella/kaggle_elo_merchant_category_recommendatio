#!/usr/bin/env python

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
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

from joblib import Parallel, delayed

def train_fold(val_idx,trn_data,val_data,**param):
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    return val_idx, clf
    
    

def train_function(**kwargs):

    global predictions
    global oof
    predictions[:] = 0.
    oof[:] = 0.
    param = copy(default_param)
    param.update(kwargs)
    print(param)
    
    int_params = ["num_leaves","min_child_samples","max_depth","min_data_in_leaf"]
    for par in int_params:
        param[par] = int(param[par])
    
    folds = KFold(n_splits=5, shuffle=True, random_state=15)
    start = time.time()
    feature_importance_df = pd.DataFrame()
    
    def get_data(trn_idx, val_idx):
        trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
        val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

        return [trn_data,val_data]

    results = Parallel(n_jobs=5,verbose=10)(delayed(train_fold)(val_idx,*get_data(trn_idx, val_idx),**param)    
                                  for (trn_idx, val_idx) in folds.split(train.values, target.values) )

    ## results  = [ train_fold(val_idx,*get_data(trn_idx, val_idx),**param)for (trn_idx, val_idx) in folds.split(train.values, target.values) ]
    
    for val_idx, clf in results:
        oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
            
    
        ### for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        ###     print("fold n°{}".format(fold_))
        ###     trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
        ###     val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

        ### ### num_round = 10000
        ### ### clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1500, early_stopping_rounds = 200)
        ### ### oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        ### 
        ### fold_importance_df = pd.DataFrame()
        ### fold_importance_df["feature"] = features
        ### fold_importance_df["importance"] = clf.feature_importance()
        ### fold_importance_df["fold"] = fold_ + 1
        ### feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        ### 
        predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
        
    ##   print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
    return -mean_squared_error(oof, target)**0.5
 
 

train = pd.read_hdf("../input/train_elo_world_nacat.hd5","train")
test = pd.read_hdf("../input/train_elo_world_nacat.hd5","test")

target = pd.read_csv('../input/train.csv.zip',usecols=['target'],squeeze=True)
predictions = np.zeros(test.shape[0])
oof = np.zeros(train.shape[0])

features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]
print(categorical_feats)

## assert(0)

default_param = {'num_leaves': 100,
                 'min_data_in_leaf': 30, 
                 'objective':'regression',
                 'max_depth': 6,
                 'learning_rate': 0.005,
                 "min_child_samples": 20,
                 "boosting": "gbdt",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9 ,
                 "bagging_seed": 11,
                 "metric": 'rmse',
                 "lambda_l1": 0.1,
                 "verbosity": -1,
                 "device" : "gpu"
}


train_function()

### optimizer = BayesianOptimization(
###     f=train_function,                                                                                                                                                                     
###     pbounds={'num_leaves': (50,200),
###              'min_data_in_leaf': (5,50), 
###              'max_depth': (4,10),
###              'learning_rate': (0.001,0.01),
###              "min_child_samples": (5,30),
###              "feature_fraction": (0.7,1.),
###              "bagging_fraction": (0.7,1.),
###              "lambda_l1": (0.05,0.3),
###     },                                                                                                                                                                                   
###     verbose=2,
###     random_state=23456,
### )
### 
### 
### if os.path.exists("./logs_lgb_continue.json"):
###     with open("./logs_continue.json") as fin:    
###         for point_str in fin.readlines():
###             point = json.loads(point_str)
###             optimizer.register(point["params"],point["target"])
### 
### logger = JSONLogger(path="./logs_lgb.json")
### optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
### 
### 
### print("before starting maximum --- ", optimizer.max)
### 
### optimizer.maximize(init_points=5,
###                    n_iter=25,kappa=5.)
### 
### 
### print("after search maximum --- ", optimizer.max)
### 
### train_function(**(optimizer.max["params"]))

sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submit-lgb-nacat.csv", index=False)

valid_df = pd.DataFrame({"card_id":train["card_id"].values})
valid_df["target"] = oof
valid_df.to_csv("valid-lgb-nacat.csv", index=False)


