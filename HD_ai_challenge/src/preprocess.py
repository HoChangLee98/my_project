import bisect
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class PreProcessor:
    def __init__(self, categorical_feature:list, minmaxscale_feature:list):
        self.categorical_feature = categorical_feature
        self.minmaxscale_feature = minmaxscale_feature

    def preprocess(self, df, method:str='area_mean', mean_values=None, ship_mean_values=None):
        df = self.nan_process(df, method=method, mean_values=mean_values, ship_mean_values=ship_mean_values)    
        df = self.date_process(df)
        
        return df
    
    def transform(self, df, encoder:dict, scaler:object):
        df = self.categorical_process_transform(df, encoder)
        df = self.minmaxscale_process_transform(df, scaler)

        return df

    # def replace_continuous_zeros_with_nan(self, input_list):
    #     result = []
    #     consecutive_zeros = 0
        
    #     for value in input_list:
    #         if value == 0:
    #             consecutive_zeros += 1
    #             if consecutive_zeros <= 2:
    #                 result.append(value)  # 연속된 최대 2개의 0은 유효한 값으로 처리
    #             else:
    #                 result.append(None)  # 연속된 3개 이상의 0은 결측치로 처리
    #         else:
    #             consecutive_zeros = 0  # 0 아닌 값이 나오면 연속된 0 카운트 초기화
    #             result.append(value)
        
    #     return result


    def nan_process(self, df, mean_values=None, ship_mean_values=None, method:str='area_mean'):
        '''결측치를 처리하는 함수 

        Args:
            drop: 
                True일 경우 모두 제거
                False일 경우 결측치 대체
        '''
    
        if method == 'drop':
            df = df.drop(columns=['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN'])
            df = df.dropna(axis=0)
        
        elif method == 'mean':
            df[['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']] = df[['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].fillna(mean_values)            
        
        elif method == 'area_mean':
            
            for po in mean_values['po'].keys():
                df.loc[df['ARI_PO'] == po, ['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']] = df.loc[df['ARI_PO'] == po, ['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].fillna(mean_values['po'][po])

            if len(df[df[['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].isna().any(axis=1)]) != 0:
                nan_data = df[df[['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].isna().any(axis=1)].copy()
                nan_data_co_list = nan_data['ARI_CO'].unique()
                for co in nan_data_co_list:
                    df[(df['ARI_CO'] == co) & (df[['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].isna().any(axis=1))] = df[(df['ARI_CO'] == co) & (df[['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].isna().any(axis=1))].fillna(mean_values['co'][co])
            
            ## 'BBREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH' 평균값 처리
            df[['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH']] = df[['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH']].fillna(ship_mean_values)            
                    
        return df
    
    def nan_mean_fit(self, df):
        nan_replace_mean = df[['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].apply(lambda x : np.mean(x))

        return nan_replace_mean

    def nan_area_mean_fit(self, df):    
        ARI_CO_uniq = df['ARI_CO'].unique()
        ARI_PO_uniq = df['ARI_PO'].unique()

        co_mean_value = {}
        po_mean_value = {}

        for co in ARI_CO_uniq:
            co_mean_value[co] = df.loc[df['ARI_CO']==co, ['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].apply(lambda x : np.mean(x))
            
        for po in ARI_PO_uniq:
            po_mean_value[po] = df.loc[df['ARI_PO']==po, ['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN']].apply(lambda x : np.mean(x))
            
        area_mean_value = {'co' : co_mean_value, 'po' : po_mean_value}        
        
        return area_mean_value
    
    def nan_ship_fit(self, df):
        nan_ship_mean = df[['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH']].apply(lambda x : np.mean(x))

        return nan_ship_mean
            

    def categorical_process_fit(self, df):
        encoder = {}

        for cat_feature in self.categorical_feature:
            le = LabelEncoder()
            le.fit(df[cat_feature].astype(str))
            encoder[cat_feature] = le 

        return encoder


    def categorical_process_transform(self, df, encoder:dict):
        for cat_feature in self.categorical_feature:
            le = encoder[cat_feature]
            le_classes_set = set(le.classes_)
            df[cat_feature] = df[cat_feature].map(lambda s: '-1' if s not in le_classes_set else s)

            le_classes = le.classes_.tolist()
            bisect.insort_left(le_classes, '-1')
            le.classes_ = np.array(le_classes)

            df[cat_feature] = le.transform(df[cat_feature].astype(str))
            df[cat_feature] = df[cat_feature].astype('category')

        return df
    
    def minmaxscale_process_fit(self, df):
        scaler = MinMaxScaler()
        scaler = scaler.fit(df[self.minmaxscale_feature].to_numpy())

        return scaler
    
    def minmaxscale_process_transform(self, df, scaler:object):
        df[self.minmaxscale_feature] = scaler.transform(df[self.minmaxscale_feature].to_numpy())
        
        return df

    def date_process(self, df):
        '''날짜 데이터 처리
        `ATA` 변수 날짜 처리
        
        '''
        df['ATA'] = pd.to_datetime(df['ATA'])
        
        df['year'] = df['ATA'].dt.year
        df['month'] = df['ATA'].dt.month
        df['day'] = df['ATA'].dt.day
        df['hour'] = df['ATA'].dt.hour
        # df['minute'] = df['ATA'].dt.minute
        # df['weekday'] = df['ATA'].dt.weekday

        df = df.drop(columns=["ATA"])

        return df
    

