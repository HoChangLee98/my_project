import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm


class Validation:
    def __init__(self):
        
    
    # def kfold(self, X, y, n_splits:int, shuffle:bool, random_state:int=None):
    #     '''
    #     ** shuffle이 True 일 경우 random_state 설정 필요!!
    #     '''
        
    #     if not shuffle:
    #         random_state = None
        
    #     kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
    #     total_y_val = []
    #     total_y_val_pred = []
        
    #     for index_train, index_valid in kf.split(X):
    #         X_tr, y_tr = X.loc[index_train], y[index_train]
    #         X_val, y_val = X.loc[index_valid], y[index_valid]

    #         model = lightgbm.LGBMRegressor()
    #         model.fit(X_tr, y_tr, eval_metric="rmse", eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    #         y_val_pred = model.predict(X_val)
    #         each_fold_rmse = mean_squared_error(y_true=y_val, y_pred=y_val_pred, squared=False)
    #         scores.append(each_fold_rmse)
    #         # print(f"     Train Set len: {len(index_train)}")
    #         # print(f"     Validation Set len: {len(index_valid)}")
    #         # print(f"     Rmse: {each_fold_rmse}")
            
    #     score = np.mean(scores)
    #     # print("-----------------------------------")
    #     # print(f"Mean Rmse: {score}")
        
    #     return score    
    
    
    def custom_walkforwad_by_case(self, X, y, n_splits, valid_size):
        train_size = len(y) // n_splits
        
        total_y_val = []
        total_y_val_pred = []
        
        for i in range(n_splits):
            s_block, e_block = i * train_size, (i+1) * train_size
            
            if i == n_splits:
                valid_size = len(y) - e_block
        
            X_tr, y_tr = X[s_block:e_block], y[s_block:e_block]
            X_val, y_val = X[e_block:e_block + valid_size], y[e_block:e_block + valid_size]

            model = lightgbm.LGBMRegressor()
            model.fit(X_tr, y_tr, eval_metric="rmse", eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
            y_val_pred = model.predict(X_val)
            
            total_y_val.append(y_val)
            total_y_val_pred.append(y_val_pred)
        
        return total_y_val, total_y_val_pred
        
        
    def custom_walkforwad_increasing_by_case(self, X, y, n_splits, valid_size):
        train_size = len(y) // n_splits
        
        total_y_val = []
        total_y_val_pred = []
        
        for i in range(n_splits):
            e_block = (i+1) * train_size
            
            if i == n_splits:
                valid_size = len(y) - e_block

            X_tr, y_tr = X[0:e_block], y[0:e_block]
            X_val, y_val = X[e_block:e_block + valid_size], y[e_block:e_block + valid_size]

            model = lightgbm.LGBMRegressor()
            model.fit(X_tr, y_tr, eval_metric="rmse", eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
            y_val_pred = model.predict(X_val)
            
            total_y_val.append(y_val)
            total_y_val_pred.append(y_val_pred)
        
        return total_y_val, total_y_val_pred
    
    