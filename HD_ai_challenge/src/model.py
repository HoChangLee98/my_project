import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import after_process

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error


class ClassificationModel:
    def __init__(
        self, 
        X_train:pd.DataFrame=None, 
        y_train:pd.Series=None,
        X_valid:pd.DataFrame=None, 
        y_valid:pd.Series=None,
        classifier_name:str=None,
        # classifier_params:dict=None
        ):
        self.X_train = X_train
        self.y_train_binary = y_train["CI_HOUR"].apply(lambda x : 1 if x != 0 else x).astype('int')
        self.X_valid = X_valid 
        self.y_valid_binary = y_valid["CI_HOUR"].apply(lambda x : 1 if x != 0 else x).astype('int')
        self.classifier_name = classifier_name
        # self.classifier_params = self.classifier_params
        
    def fit(self):
        if self.classifier_name is not None:
            if self.classifier_name == "logistic":
                cls = LogisticRegression(
                    random_state=0, 
                    class_weight='balanced', 
                    max_iter=100, 
                    multi_class='ovr', 
                    verbose=0
                    )

            elif self.classifier_name == "lightgbm":
                cls = LGBMClassifier(
                    force_col_wise=True,
                    objective='binary',
                    class_weight='balanced',
                    is_unbalance=True,
                    seed=0
                    )
                
            fitted_classifier = cls.fit(self.X_train, self.y_train_binary)
            y_pred_binary = fitted_classifier.predict(self.X_valid)
            print(f"    ##{self.classifier_name}##")
            print("accuracy : ", accuracy_score(y_true=self.y_valid_binary, y_pred=y_pred_binary))
            print("f1 : ", f1_score(y_true=self.y_valid_binary, y_pred=y_pred_binary))
            print("precision : ", precision_score(y_true=self.y_valid_binary, y_pred=y_pred_binary))
            print("recall : ", recall_score(y_true=self.y_valid_binary, y_pred=y_pred_binary))
            
            return fitted_classifier
    
def output_index(df:pd.Series, classifier:object=None):
    if classifier is not None:
        binary_target_pred = pd.Series(classifier.predict(df))
        print("Length of None Zero Target : ", sum(binary_target_pred))
        
        none_zero_index = binary_target_pred.loc[binary_target_pred != 0].index
        
        return none_zero_index       
    

class RegressionModel:
    def __init__(
            self, 
            model_params:dict,
            categorical_feature:list,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            model_name:str,
            X_valid:pd.DataFrame=None,
            y_valid:pd.Series=None,
            ):
        self.model_params = model_params
        self.categorical_feature = categorical_feature
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.model_name = model_name

    
    def fit(self):
        if self.model_name == 'lightgbm':
            model = self.lightgbm_fit()

        elif self.model_name == 'randomforest':
            model = self.randomforest_fit()
        
        elif self.model_name == 'catboost':
            model = self.catboost_fit()
        
        elif self.model_name == 'xgboost':
            model = self.xgboost_fit()

        return model
    

    def lightgbm_fit(self):

        model = lgb.LGBMRegressor(
            **self.model_params, 
            seed = 0,
            force_col_wise = True,
            objective = 'regression_l1',
            metric = 'l1', 
            verbose = -1, 
            early_stopping_rounds=100, 
            categorical_feature = self.categorical_feature,                             
            )
        
        model = model.fit(
            self.X_train, 
            self.y_train, 
            eval_set=[(self.X_valid, self.y_valid)], 
            )
                           
        return model
    
    def randomforest_fit(self):
        model = RandomForestRegressor(
            **self.model_params, 
            criterion = "absolute_error",
            random_state = 0,
            verbose = False,
            )
        model = model.fit(X=self.X_train, y=self.y_train)

        return model

    def catboost_fit(self):
        model = CatBoostRegressor(**self.model_params)
        model = model.fit(X=self.X_train, y=self.y_train, 
                          eval_set=(self.X_valid, self.y_valid), 
                          cat_features=self.categorical_feature, 
                          use_best_model=True, 
                          verbose=False, 
                          verbose_eval=False, 
                          early_stopping_rounds=100)
        return model

    def xgboost_fit(self):        
        model = XGBRegressor(
            **self.model_params, 
            verbosity = 0,
            objective = 'reg:absoluteerror', 
            eval_metric = 'mae', 
            seed = 0, 
            enable_categorical = True,
            tree_method = 'hist'
            )
        
        model = model.fit(
            self.X_train, 
            self.y_train, 
            eval_set=[(self.X_valid, self.y_valid), (self.X_train, self.y_train)], 
            verbose=False, 
            early_stopping_rounds=100,
            )
        
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
        if model_name == 'lightgbm':
            study.optimize(lambda trial : self.objective_lightgbm(trial), n_trials=n_trials)
        elif model_name == 'randomforest':
            study.optimize(lambda trial : self.objective_randomforest(trial), n_trials=n_trials)
        elif model_name == 'catboost':
            study.optimize(lambda trial : self.objective_catboost(trial), n_trials=n_trials)
        elif model_name == 'xgboost':
            study.optimize(lambda trial : self.objective_xgboost(trial), n_trials=n_trials)

        return study

    def objective_lightgbm(self, trial: Trial):
        params = {
            'num_iteration' : trial.suggest_int('num_iteration', 100, 1000),
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.9),
            'num_leaves' : trial.suggest_int('num_leaves', 2, 100),
            # 'nthread' : -1,
            'seed' : 0,
            'force_col_wise' : True,
            'max_depth' : trial.suggest_int('max_depth', 1, 10),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 300),
            'early_stopping_round' : 100,
            # 'lambda_l1' : trial.suggest_loguniform('lambda_l1', 0.0001, 10),
            # 'lambda_l2' : trial.suggest_loguniform('lambda_l2', 0.0001, 10),
            'categorical_feature' : self.categorical_feature,
            'objective' : 'regression_l1',
            'metric' : 'l1', 
            'verbose' : -1     
        }
       
        model = lgb.LGBMRegressor(**params)
        model = model.fit(
            self.X_train, 
            self.y_train, 
            eval_set=[(self.X_valid, self.y_valid)], 
            )
                
        # 모델 성능 확인
        score = mean_absolute_error(after_process(model.predict(self.X_valid)), self.y_valid)
        
        return score

    def objective_randomforest(self, trial: Trial):
        params = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 10000), ## (100, 10000)
            'criterion' : "absolute_error",
            'max_depth' : trial.suggest_int('max_depth', 1, 10), 
            'min_samples_split' : trial.suggest_int('min_samples_split', 2, 30), 
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 100),
            'random_state' : 0,
            'verbose' : False,
            'max_samples' : trial.suggest_int('max_samples', 3, 1000)
        }
                    
        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)

        score = mean_absolute_error(after_process(model.predict(self.X_valid)), self.y_valid)

        return score
    
    def objective_catboost(self, trial: Trial):
        params = {
            'one_hot_max_size' : 20, 
            'iterations' : trial.suggest_int('iterations', 100, 1000),
            # 'use-best-model' : True, 
            'eval_metric' : 'MAE', 
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
            'depth' : trial.suggest_int('depth', 3, 16), 
            'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0.0, 1.0),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 60), 
            # 'max_leaves' : trial.suggest_int('max_leaves', 1, 60), 
        }

        model = CatBoostRegressor(**params)
        model = model.fit(X=self.X_train, y=self.y_train, 
                          eval_set=(self.X_valid, self.y_valid), 
                          cat_features=self.categorical_feature, 
                          use_best_model=True, 
                          verbose=False, 
                        #   verbose_eval=False, 
                          early_stopping_rounds=100)        
    
    def objective_xgboost(self, trial: Trial):
        params = {
            # 'num_boost_round' : trial.suggest_int('num_boost_round', 100, 1000), 
            # 'booster' : 'gblinear', 
            'verbosity' : 0,
            'eta' : trial.suggest_loguniform('eta', 0.01, 0.9),
            'gamma' : trial.suggest_float('gamma', 0, 10), 
            'max_depth' : trial.suggest_int('max_depth', 1, 100), 
            'min_child_weight' : trial.suggest_loguniform('min_child_weight', 0.5, 1), # 0,500 
            # 'max_delta_step' : trial.suggest_int('max_delta_step', 1, 50), # 1, 50
            # 'subsample' : trial.suggest_loguniform('subsample', 0.03, 0.5), 
            'colsample_bytree' : trial.suggest_loguniform('colsample_bytree', 0.5, 1), 
            # 'colsample_bylevel' : trial.suggest_loguniform('colsample_bylevel', 0.0001, 1), 
            # 'colsample_bynode' : trial.suggest_loguniform('colsample_bynode', 0.0001, 1),   
            'lambda' : trial.suggest_float('lambda', 0.0, 10.0),
            'alpha' : trial.suggest_float('alpha', 0.0, 10.0),
            # 'num_parallel_tree' : trial.suggest_int('num_parallel_tree', 2, 100), 
            'max_cat_to_onehot' : trial.suggest_int('max_cat_to_onehot', 1, 10), 
            'objective' : 'reg:absoluteerror', 
            'eval_metric' : 'mae', 
            'seed' : 0, 
            # 'num_round' : trial.suggest_int('num_round', 100, 1000), 
            'n_estimaters' : trial.suggest_int('n_estimaters', 100, 1000), # 100, 10000
            'enable_categorical' : True,
            'tree_method' : 'hist'
         }

        # features = self.X_train.columns.tolist()
        # train_data = pd.concat([self.X_train, self.y_train], axis=1)
        # valid_data = pd.concat([self.X_valid, self.y_valid], axis=1)
        
        model = XGBRegressor(**params)
        model = model.fit(
            self.X_train, 
            self.y_train, 
            eval_set=[(self.X_valid, self.y_valid), (self.X_train, self.y_train)], 
            verbose=False, 
            early_stopping_rounds=100, 
            )
        
        score = mean_absolute_error(after_process(model.predict(self.X_valid)), self.y_valid)

        return score
