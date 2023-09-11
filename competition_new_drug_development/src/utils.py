import os
import pandas as pd
import numpy as np
from typing import Tuple

import pickle
import random
from datetime import datetime
from sklearn.metrics import mean_squared_error



def dataloader(target_name:str=None, folder_path:str=None)->tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    '''데이터를 불러오고 train set의 X값과 y값을 분리해주며 test set의 X값을 돌려주는 함수

    '''
    if folder_path == None:
        X_train = pd.read_csv(f"../data/X_train_{target_name}.csv")
        X_valid = pd.read_csv(f"../data/X_valid_{target_name}.csv")
        y_train = pd.read_csv(f"../data/y_train_{target_name}.csv")
        y_valid = pd.read_csv(f"../data/y_valid_{target_name}.csv")
        
        test = pd.read_csv("../data/test.csv")
        test = test.drop(columns="id")

        return X_train, X_valid, y_train, y_valid, test

    else:
        train = pd.read_csv(f"{folder_path}/train.csv")
        test = pd.read_csv(f"{folder_path}/test.csv")
        train = train.drop(columns="id") ; test = test.drop(columns="id")

        X_train = train.drop(columns=['MLM', 'HLM'], axis=1)
        y_train_MLM = train[['MLM']]  
        y_train_HLM = train[['HLM']]  

        return X_train, y_train_MLM, y_train_HLM, test


def load_pickle(file_name, save_path:str="./pickles"):
    with open(f"{save_path}/{file_name}.pkl", "rb") as f:
        file = pickle.load(f)        

    return file


def save_pickle(file, file_name, save_path:str="./pickles"):
    # settime = datetime.now().strftime("%y%m%d%H%M%S")
    # file_name = f"{file_name}_{settime}.pkl"

    with open(f"{save_path}/{file_name}.pkl", "wb") as f:
        pickle.dump(file, f)        


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

