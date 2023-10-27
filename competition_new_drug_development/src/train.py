from utils import *
from feature_engineering import *
from model import *

import lightgbm
from lightgbm import LGBMRegressor

def main(target_name:str, opt_tune:bool=False, eval:bool=False):

    if opt_tune == True:
        ## load data
        X_train, X_valid, y_train, y_valid, test = dataloader(target_name=target_name)

        ## feature engineering
        fe = FeatureEngineer(X_train=X_train, X_test=X_valid)
        X_train, X_valid = fe.drop_cat_feature()
 
        print("=====Start Optuna=====")
        get_optuna_params(X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
        print("=====End Oputna=====")
   
    elif opt_tune == False:
        ## load data
        X_train, X_valid, y_train, y_valid, test = dataloader(target_name=target_name)

        ## feature engineering
        fe = FeatureEngineer(X_train=X_train, X_test=X_valid)
        X_train, X_valid = fe.drop_cat_feature()

        ## modeling
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_valid, y_valid)

        rmse = root_mean_square_error(y_valid, y_val_pred)




