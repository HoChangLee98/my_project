import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    def __init__(self, X_train=None, X_test=None):
        self.X_train = X_train
        self.X_test = X_test

    def drop_cat_feature(self, df):
        df_drop = df.drop(columns="SMILES")
        
        return df_drop        
