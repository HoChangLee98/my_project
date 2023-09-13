import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple


class FeatureEngineer:
    def __init__(self, X_train=None, X_test=None):
        self.X_train = X_train
        self.X_test = X_test
        

    def feature_engineering_process(
            self, 
            df,
            minmaxscaler:object=None, 
            labelencoder:object=None,
            minmax_columns:list=None
            )->tuple[pd.DataFrame, pd.Series] or pd.DataFrame:
        
        
            # X = self.drop_cat_feature(self.X_test)
            df = self.transform_labelencoder(df, encoder=labelencoder)
            df = self.transform_minmaxscaler(df, scaler=minmaxscaler, columns=minmax_columns)

            return df
           

    def drop_cat_feature(self, df):
        df_drop = df.drop(columns="SMILES")
        
        return df_drop        


    def fit_minmaxscaler(self, df:pd.DataFrame, columns:list)->object:
        """주어진 데이터를 MinMaxScaler 학습
        
        Args: 
            df: MinMaxScaling 하고자하는 학습 데이터 X_train
        """
        scaler = MinMaxScaler()
        scaler.fit(df[columns].to_numpy())

        return scaler

    def fit_labelencoder(self, df:pd.DataFrame, columns:list=["SMILES"])->object:
        """주어진 데이터의 범주형 변수 labelencoding

        Args:
            df: X_train 과 같은 학습데이터
            columns: 범주형 변수
        """
        encoder = LabelEncoder()
        encoder.fit(df[columns].values)

        return encoder

    def transform_minmaxscaler(self, df:pd.DataFrame, scaler:object, columns:list)->pd.DataFrame:
        df[columns] = scaler.transform(df[columns].to_numpy())

        return df

    def transform_labelencoder(self, df:pd.DataFrame, encoder:object, columns:list=["SMILES"])->pd.DataFrame:
        df[columns] = encoder.transform(df[columns].values).reshape(-1,1)

        return df
