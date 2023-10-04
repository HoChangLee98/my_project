import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error


class Model:
    def __init__(
            self, 
            model_params:dict,
            categorical_feature:list,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            X_valid:pd.DataFrame=None,
            y_valid:pd.Series=None,
            model_name:str='lgboost'
            ):
        self.model_params = model_params
        self.categorical_feature = categorical_feature
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.model_name = model_name

    
    def fit(self):
        if self.model_name == 'lgboost':
            model = self.lgboost_fit()

        if self.model_name == 'randomforest':
            model = self.randomforest_fit()

        return model
    

    def lgboost_fit(self):
        train_data = lgb.Dataset(
            data=self.X_train, 
            label=self.y_train, 
            categorical_feature=self.categorical_feature,
            params={'verbose': -1}
            )
        
        valid_data = lgb.Dataset(
            data=self.X_valid,
            label=self.y_valid,
            categorical_feature=self.categorical_feature,
            params={'verbose': -1}
            )
        
        params = self.model_params

        model = lgb.train(params, train_data, valid_sets=[valid_data])

        return model
    
    def randomforest_fit(self):
        model = RandomForestRegressor(**self.model_params)
        model = model.fit(X=self.X_train, y=self.y_train)

        return model


class OptunaProcessor:
    def __init__(
            self, 
            X_train, 
            y_train, 
            X_valid,
            y_valid, 
            categorical_feature
            ):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.categorical_feature = categorical_feature

    def run_optuna(self, n_trials:int, model_name:str):
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        if model_name == 'lgboost':
            study.optimize(lambda trial : self.objective_lgboost(trial), n_trials=n_trials)
        elif model_name == 'randomforest':
            study.optimize(lambda trial : self.objective_randomforest(trial), n_trials=n_trials)

        return study

    def objective_lgboost(self, trial: Trial):
        params = {
            'num_iteration' : trial.suggest_int('num_iteration', 100, 1000),
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
            'num_leaves' : trial.suggest_int('num_leaves', 2, 100),
            # 'nthread' : -1,
            'seed' : 0,
            'force_col_wise' : True,
            'max_depth' : trial.suggest_int('max_depth', 3, 50),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 300),
            'early_stopping_round' : 100,
            'lambda_l1' : trial.suggest_loguniform('lambda_l1', 0.0001, 10),
            'lambda_l2' : trial.suggest_loguniform('lambda_l2', 0.0001, 10),
            # 'categorical_feature' : self.categorical_feature,
            'objective' : 'regression_l1',
            'metric' : 'l1'     
        }
        
        # 학습 모델 생성
        train_data = lgb.Dataset(
            data=self.X_train, 
            label=self.y_train, 
            categorical_feature=self.categorical_feature,
            params={'verbose': -1}
            )

        valid_data = lgb.Dataset(
            data=self.X_valid,
            label=self.y_valid,
            categorical_feature=self.categorical_feature,
            params={'verbose': -1}
            )
        
        model = lgb.train(params, train_data, valid_sets=[valid_data])
        
        # 모델 성능 확인
        score = mean_absolute_error(model.predict(self.X_valid), self.y_valid)
        
        return score

    def objective_randomforest(self, trial: Trial):
        params = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'criterion' : "absolute_error",
            'max_depth' : trial.suggest_int('max_depth', 3, 30), 
            'min_samples_split' : trial.suggest_int('min_samples_split', 2, 30), 
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 30),
            'random_state' : 0,
            'verbose' : False,
            'max_samples' : trial.suggest_int('max_samples', 3, 100)
        }

        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)

        score = mean_absolute_error(model.predict(self.X_valid), self.y_valid)

        return score