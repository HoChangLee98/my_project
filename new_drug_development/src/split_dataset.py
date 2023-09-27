import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_valid(file_name:str="../data"):
    train = pd.read_csv(f"{file_name}/train.csv")
    test = pd.read_csv(f"{file_name}/test.csv")
    
    train = train.drop(columns="id")
    test = test.drop(columns="id")
    
    X_train = train[train.columns.difference(["MLM", "HLM"]).tolist()]
    y_train_MLM = train['MLM']
    y_train_HLM = train['HLM']

    X_train_MLM, X_valid_MLM, y_train_MLM, y_valid_MLM = train_test_split(X_train, y_train_MLM, test_size=0.3, random_state=0)
    X_train_HLM, X_valid_HLM, y_train_HLM, y_valid_HLM = train_test_split(X_train, y_train_HLM, test_size=0.3, random_state=0)

    X_train_MLM.to_csv(f"{file_name}/X_train_MLM.csv", index=False)
    X_valid_MLM.to_csv(f"{file_name}/X_valid_MLM.csv", index=False)
    y_train_MLM.to_csv(f"{file_name}/y_train_MLM.csv", index=False)
    y_valid_MLM.to_csv(f"{file_name}/y_valid_MLM.csv", index=False)

    X_train_HLM.to_csv(f"{file_name}/X_train_HLM.csv", index=False)
    X_valid_HLM.to_csv(f"{file_name}/X_valid_HLM.csv", index=False)
    y_train_HLM.to_csv(f"{file_name}/y_train_HLM.csv", index=False)
    y_valid_HLM.to_csv(f"{file_name}/y_valid_HLM.csv", index=False)


split_train_valid()