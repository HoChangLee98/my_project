import os
import random
import pickle
import numpy as np
import pandas as pd

def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def reset_data(X, y):
    y = y.loc[X.index]
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y

def cat_idx(X):
    cat_idx = np.where(X.dtypes == object)[0].tolist()
    
    return cat_idx

def load_dataset(mode:str='inference'):
    if mode == 'train':
        X_train = pd.read_csv("../data/X_train.csv")
        y_train = pd.read_csv("../data/y_train.csv")
        X_valid = pd.read_csv("../data/X_valid.csv")
        y_valid = pd.read_csv("../data/y_valid.csv")

        return X_train, y_train, X_valid, y_valid
    
    else:
        X_test = pd.read_csv("../data/X_test.csv")

        return X_test        

def save_pickle(file, file_name:str, path:str):
    os.makedirs(f"{path}", exist_ok=True)
    with open(f"{path}/{file_name}.pkl", 'wb') as f:
        pickle.dump(file, f)

    print(f"Done {file_name} Save!")

def load_pickle(file_name:str, path:str):
    with open(f"{path}/{file_name}.pkl", 'rb') as f:
        file = pickle.load(f)

    return file

def generate_submission_file(y_pred, file_name:str, path:str, classification_method=None, none_zero_index=None):
    
    os.makedirs(f"{path}", exist_ok=True)
    submission = pd.read_csv("../submission/sample_submission.csv")
    if classification_method == 'True':
        submission['CI_HOUR'] = 0
        submission.loc[none_zero_index, 'CI_HOUR'] = y_pred
    else: 
        submission['CI_HOUR'] = y_pred

    submission.to_csv(f"{path}/{file_name}.csv", index=False) 
    
    print(submission)
    print("Done Generating Submission!")

def after_process(y_pred):
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.apply(lambda x : 0 if x < 1 else np.round(x, 5))
    
    return y_pred.to_numpy() 