import pandas as pd


class DataLoader:
    '''X_train, y_train_supply, y_train_price csv 파일을 불러오는 함수
    단, item과 corporation을 기준으로 분류하여 dict에 저장하여 dict를 준다.

    '''
    def __init__(self, file_path:str="../data"):
        self.file_path = file_path
    
    def dataloader(self, file_name:str, method:str="reg"):
        df = pd.read_csv(f"{self.file_path}/{file_name}.csv")
        if file_name in ["X_train", "X_test"]:
            df['timestamp'] = pd.to_datetime(df)
        
        df_dict = self.preprocessor(df, method=method)

        return df_dict
        
    def preprocessor(self, df, method:str): 
        
        if method == "reg":
            df['year'] = df['timestamp'].dt.year()
            df['month'] = df['timestamp'].dt.month()
            df['day'] = df['timestamp'].dt.day()
            df['day_name'] = df['timestamp'].dt.day_name()
            

        elif method == "time":
            df_dict = {}
            feature = list(df.columns)[:-3]
            for item_ in df["item"].unique():
                df_dict[item_] = {}
                for cor_ in df["corporation"].unique():
                    df_dict[item_][cor_] = {}
                    df_dict[item_][cor_]["J"] = df.loc[(df['item'] == item_) & (df['corporation'] == cor_) & (df['location'] == "J"), feature]
                    df_dict[item_][cor_]["S"] = df.loc[(df['item'] == item_) & (df['corporation'] == cor_) & (df['location'] == "S"), feature]
            
            return df_dict
    
    