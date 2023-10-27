import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def drop_cat_feature(self):
        X_train_drop = self.X_train.drop(colums="SMILES")
        X_test_drop = self.X_test.drop(colums="SMILES")

        return X_train_drop, X_test_drop
        
