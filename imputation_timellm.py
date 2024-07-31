# part CSV 한개 씩 불러오기
# obs, phar 각각 같은 part에는 같은 환자가 소속되어 있음
import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
import warnings
from imp import reload
import outlier_removal
reload(outlier_removal)

pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandasql as psql

local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'

# col = ['Lactate','Milrinone','MAP','Fluid Bolus','Platelet count','ABPs','Bilirubin','FiO2','SpO2',
#       'Time_since_ICU_admission','Dobutamine','Temperature','PEEP','ABPd','Urine_output','Theophyllin',
#       'Antibiotics','INR','HR','pH','Creatinine','patientid','PaO2','Respiratory_rate']

if not os.path.exists(local+"tabular_records/csv_imputation_10/"):
            os.makedirs(local+"tabular_records/csv_imputation_10/")

def Imputation(part_list):
    print("[ EXCUTING DATA IMPUTATION ]")
    for parts in part_list:
        print(f'Part {parts} processing......')

        part = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_tab_10/part-{parts}.csv')
        
        result = pd.DataFrame()

        for stay in part.patientid.unique():
            
            current_stay = psql.sqldf(f"SELECT * FROM part WHERE patientid = {stay};", locals())
            current_stay.rename(columns={"Platelet count":"Platelet_count"}, inplace=True)
                
            required_columns = ['Dobutamine', 'Milrinone', 'Vasopressin', 'Norepinephrine']

      
            for v in required_columns:
                if v not in current_stay.columns:
                    current_stay[v] = 0

            current_stay[required_columns] = current_stay[required_columns].fillna(0).reset_index(drop=True)

            current_stay['Vasopressin'] = (current_stay['Vasopressin'] > 0).astype(int)
            current_stay['Norepinephrine'] = (current_stay['Norepinephrine'] > 0).astype(int)
            current_stay['Dobutamine'] = (current_stay['Dobutamine'] > 0).astype(int)
            current_stay['Milrinone'] = (current_stay['Milrinone'] > 0).astype(int)
            current_result = outlier_removal.GETOUT_ALL(current_stay)
            result = pd.concat([result, current_result], axis = 0)
            
            
        dir = f'tabular_records/csv_imputation_10/part-{parts}.csv'
        
        result['vaso'] = result['Dobutamine'] + result['Milrinone'] + result['Vasopressin'] + result['Norepinephrine']
        result['vasopressor'] = result['vaso'].apply(lambda x: 1 if not pd.isna(x) and x > 0 else 0)
        
        result = result.drop(['vaso', 'Dobutamine', 'Milrinone', 'Vasopressin', 'Norepinephrine'], axis = 1)
        
        result.to_csv(local+dir,index=False)
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")
            
