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



# 3. Train with early stopping
def train_lgbm(model, X_train, y_train, X_val, y_val, early_stopping_rounds=50):
    """
    Trains the LightGBM model with early stopping to prevent overfitting.

    Early stopping monitors validation loss during training and stops
    automatically if it hasn't improved after N rounds — preventing
    the model from memorizing training data.

    Parameters:
        model                 : LGBMClassifier from get_lgbm_model()
        X_train, y_train      : training data (already SMOTE balanced)
        X_val, y_val          : validation data (never SMOTE'd)
        early_stopping_rounds : stop if no improvement after N rounds

    Returns:
        model : trained LGBMClassifier
    """
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=50)  # print metrics every 50 rounds
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=callbacks
    )

    print(f"\n[LightGBM] Training complete.")
    print(f"[LightGBM] Best iteration: {model.best_iteration_}")

    return model



# 4. Feature importance plot

import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the top N most important features from the trained
    LightGBM model.

    LightGBM calculates feature importance by counting how many
    times each feature was used to split data across all trees
    (split importance) or by how much each split improved the
    model (gain importance).

    Parameters:
        model         : trained LGBMClassifier
        feature_names : list of column names
        top_n         : how many top features to display
    """
    importance = pd.DataFrame({
        'feature'   : feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'][::-1], importance['importance'][::-1])
    plt.xlabel('Feature Importance (split count)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} LightGBM Feature Importances')
    plt.tight_layout()
    plt.savefig('lgbm_feature_importance.png')
    plt.show()

    print(f"\nTop 5 most important features:")
    print(importance.head(5).to_string(index=False))

    return importance

