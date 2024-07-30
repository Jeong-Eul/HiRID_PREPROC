import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
import warnings
pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning)

local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/'

VITAL_permitted_range = {'ABPs' : [10, 300], 'ABPd' : [10, 175], 'MAP' : [10, 258], 'HR' : [0, 300], 'PaO2' : [20, 500],
                   'FiO2' : [21, 100],'SpO2' : [10, 100], 'Respiratory_rate' : [0, 45],
                   'Temperature' : [20, 50], 'Urine_output' : [0, 1000]}

LAB_permitted_range = {'Bilirubin' : [0, 1000], 'Creatinine' : [0, 1500], 'INR' : [0, 8],
                   'Lactate' : [0, 15], 'pH' : [6.5, 7.8]}

def GETOUT(data, mode):

    print("[ EXCUTING  OUTLIER PROCESSING ]")
    
    if mode == 'V':
        permitted_range = VITAL_permitted_range.copy()
    else:
        permitted_range = LAB_permitted_range.copy()

    df = data.copy()
    num_ol = 0
    for col, range in permitted_range.items():
        
        current_view = df[~(df[col]==-100)]
        
        out_of_min = current_view[current_view[col] < range[0]].index
        out_of_max = current_view[current_view[col] > range[1]].index
        
        current_ol = len(out_of_min) + len(out_of_max)
        num_ol += current_ol
        
        df.loc[out_of_min, col] = -100
        df.loc[out_of_max, col] = -100    
        
    print('num of outlier: ', num_ol)
    
    # df.to_csv(local + 'hirid_fin.csv.gz',index=False)
    
    
    print('[ FINISH OUTLIER PROCESSING ]')
    
    return df

def GETOUT_ALL(data):
    
    print("[ EXCUTING  OUTLIER PROCESSING ]")
    
    permitted_range = {'ABPs' : [10, 300], 'ABPd' : [10, 175], 'MAP' : [10, 258], 'HR' : [0, 300], 'PaO2' : [20, 500],
                   'FiO2' : [21, 100],'SpO2' : [10, 100], 'Respiratory_rate' : [0, 45],
                   'Temperature' : [20, 50], 'Bilirubin' : [0, 1000], 'Creatinine' : [0, 1500], 'INR' : [0, 8],
                   'Lactate' : [0, 15], 'pH' : [6.5, 7.8]}

    df = data.copy()
    num_ol = 0
    for col, range in permitted_range.items():
        
        current_view = df[~(df[col]==-100)]
        
        out_of_min = current_view[current_view[col] < range[0]].index
        out_of_max = current_view[current_view[col] > range[1]].index
        
        current_ol = len(out_of_min) + len(out_of_max)
        num_ol += current_ol
        
        df.loc[out_of_min, col] = np.nan
        df.loc[out_of_max, col] = np.nan   
  
    print('[ FINISH OUTLIER PROCESSING ]')
    
    return df