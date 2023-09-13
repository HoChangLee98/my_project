import os
import pandas as pd
import numpy as np
from typing import Tuple

import pickle
import random
from datetime import datetime
from sklearn.metrics import mean_squared_error



def dataloader(target_name:str=None, folder_path:str="../data")->tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    '''데이터를 불러오고 train set의 X값과 y값을 분리해주며 test set의 X값을 돌려주는 함수

    '''
    X_train = pd.read_csv(f"{folder_path}/X_train_{target_name}.csv")
    X_valid = pd.read_csv(f"{folder_path}/X_valid_{target_name}.csv")
    y_train = pd.read_csv(f"{folder_path}/y_train_{target_name}.csv")
    y_valid = pd.read_csv(f"{folder_path}/y_valid_{target_name}.csv")
    
    test = pd.read_csv(f"{folder_path}/test.csv")
    test = test.drop(columns="id")

    return X_train, X_valid, y_train, y_valid, test


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


def outlier_remove_only_train(X_train, y_train, columns:list):
    """데이터의 이상치를 제거한 index를 돌려주는 함수
    
    학습 데이터에만 적용
    
    Args:
        X_train: 학습 데이터 셋
        y_train: 학습 데이터의 label
    """

    def outlier_IQR(data, threshold=1.5):
        q1, q3 = np.percentile(data, [25, 75]) # 1사분위수, 3사분위수 계산
        iqr = q3 - q1 # IQR 계산

        lower_bound = q1 - (threshold * iqr) # Outlier 판단 Lower Bound 계산
        upper_bound = q3 + (threshold * iqr)  #Outlier 판단 Upper Bound 계산
        
        index = []
        for i, x in enumerate(data):
            if x >= lower_bound and x <= upper_bound:
                index.append(i)

        return index 
    
    index = X_train[columns].apply(lambda x: outlier_IQR(x))
    index = list(set(index[0]) & set(index[1]))
    X_train = X_train.iloc[index, :].reset_index(drop=True)
    y_train = y_train.iloc[index]

    return X_train, y_train