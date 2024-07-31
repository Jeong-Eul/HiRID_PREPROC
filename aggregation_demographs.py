import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
import warnings
pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning)

local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/'

def P_Aggregation_Demograph(part_list):
    print("[ EXCUTING AGGREGATION & GET DEMOGRAPHS ]")
    result = pd.DataFrame()
    patient = pd.read_csv('/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/general_table.csv', usecols=['patientid', 'sex', 'age'])
    for parts in part_list:
        # print(f'Part {parts} processing......')

        part = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_exclusion_10/part-{parts}.csv')
        demo = pd.merge(part, patient, how = 'left', on='patientid')
        result = pd.concat([result, demo], axis = 0)
    
    result['sex'] = result['sex'].replace({'M' : 1, 'F':0})
    result.reset_index(drop=True, inplace=True)
    
    result = result.rename(columns = {'sex':'Sex', 'age':'Age', 'Platelet_count':'Platelet_Count', 'vasopressor':'Vasopressor'})
    print('=======')
    print('Num of obs: ', len(result))
    print('num of patient: ', result.patientid.nunique())
    print('MAN: ', len(result[result['Sex'] == 1]))
    print('WOMAN: ', len(result[result['Sex'] == 0]))
    print('=======')
    result.to_csv(local + 'hirid_shock_10min.csv.gz',index=False)
    
    print("[ FINISH AGGREGATION & GET DEMOGRAPHS ]")
    