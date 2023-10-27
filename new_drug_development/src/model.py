import pandas as pd
import numpy as np

import lightgbm
from lightgbm import LGBMRegressor

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

from sklearn.metrics import mean_squared_error

from utils import *


class Models:
    def __init__(self, model_name:str, params:dict):
        self.params = params
        self.model_name = model_name
        
    def fit(self, X_train, y_train):
        if self.model_name == "lgboost":
            model = LGBMRegressor(**self.params)
            model.fit(X_train, y_train)

        return model


## optuna
def get_optuna_params(
        model_name:str, 
        X_train, 
        y_train, 
        X_valid, 
        y_valid, 
        n_trial:int=50
        ):

    if model_name == "lgboost":
        def optuna_LGBM(trial: Trial, X_train, y_train, X_valid, y_valid):
            params = {
                "task" : "predict",
                "num_iterations" : trial.suggest_int("num_iterations", 500, 2000),
                "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
                "num_leaves" : trial.suggest_int("num_leaves", 2, 31),
                "seed" : trial.suggest_int("seed", 0, 5000),
                "max_depth" : trial.suggest_int("max_depth", 0, 10),
                # "early_stopping_round" : 100
            }

            model = Models(model_name=model_name, params=params)
            fitted = model.fit(X_train, y_train)
            y_val_pred_opt = fitted.predict(X_valid)
            rmse = mean_squared_error(y_true=y_valid, y_pred=y_val_pred_opt, squared=False)

            return rmse

    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(lambda trial : optuna_LGBM(trial, X_train, y_train, X_valid, y_valid), n_trials=n_trial)
        
    print("Best Score : ", study.best_trial.value)
    print("Best Params : ", study.best_trial.params)
    
    return study.best_trial.params