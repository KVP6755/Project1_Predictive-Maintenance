"""
Week 3: LightGBM Modeler
=========================
Initializes and configures the LGBMClassifier for imbalanced
machine failure prediction. Designed as a standalone module that
accepts any training data configuration.

Author  : Varnika Valliammai V
Branch  : varnika
File    : lgbm_model.py
"""

import lightgbm as lgb
import numpy as np
import pandas
from sklearn.model_selection import train_test_split

#1 initialize LightGBM with imbalance settings

def get_lgbm_model(scale_pos_weight = None, use_is_unbalance= False,random_state=42):

    """
    initializes a LGBMClassifier configured for imbalance data
    TWO APPROACHES FOR HANDLING DATA IMBALANCE:
    1. scale_pos_weight:
       Explicitly tells LightGBM how much more to penalize missing a faillure
       vs missing a normal reading.

    2. is_unbalanced = True:
       Lets LightGBM automatically calculate class weights internally.
       Simpler but less control.

    RETURNS:
        model : configured LGBMClassifier (not yet trained)

    """
    if use_is_unbalance:
        model=lgb.LGBMClassifier(
            objective='binary',
            is_unbalance= True,
            random_state=random_state,
            verbose=-1
        )
        print("[LightGBM] Initialized with is_unbalance=True")

    else:
        if scale_pos_weight is None:
            scale_pos_weight =9661/339

        model = lgb.LGBMClassifier(
            objective ='binary',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1
        )
        print("[LightGBM] Initialized with is_unbalanced=True")

    return model

# 2. Quick test — verify initialization works

if __name__=="__main__":
    model_a = get_lgbm_model(scale_pos_weight=28.5)
    print("Model B params: ", model_a.get_params())

    model_b = get_lgbm_model(use_is_unbalance=True)
    print("Model B params: ", model_b.get_params())

    print("LightGBM initialization working")
