import numpy as np

def make_feature(df):
    df['wind_speed'] = np.sqrt((df['U_WIND']*3.6)**2 + (df['V_WIND']*3.6)**2)
    
    return df