# part CSV 한개 씩 불러오기
# obs, phar 각각 같은 part에는 같은 환자가 소속되어 있음
import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
import warnings
pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandasql as psql

local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'

# col = ['Lactate','Milrinone','MAP','Fluid Bolus','Platelet count','ABPs','Bilirubin','FiO2','SpO2',
#       'Time_since_ICU_admission','Dobutamine','Temperature','PEEP','ABPd','Urine_output','Theophyllin',
#       'Antibiotics','INR','HR','pH','Creatinine','patientid','PaO2','Respiratory_rate']

if not os.path.exists(local+"tabular_records/csv_imputation/"):
            os.makedirs(local+"tabular_records/csv_imputation/")

def Imputation(part_list):
    print("[ EXCUTING DATA IMPUTATION ]")
    for parts in part_list:
        print(f'Part {parts} processing......')

        part = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_tab/part-{parts}.csv')
            
        result = pd.DataFrame()

        for stay in part.patientid.unique():
            
            current_stay = psql.sqldf(f"SELECT * FROM part WHERE patientid = {stay};", locals())
            current_stay.rename(columns={"Platelet count":"Platelet_count"}, inplace=True)
            
            # forward fill phase
            forward_set = current_stay[['patientid', 'Time_since_ICU_admission', 'ABPd', 'ABPs', 'FiO2', 'HR', 'MAP', 'PaO2', 'Respiratory_rate', 'SpO2', 'Temperature']]
            forward_set = forward_set.ffill()
            forward_set = forward_set.fillna(-100).reset_index(drop=True)
            forward_set['PEEP'] = current_stay['PEEP'].copy()
            # lab up down phase
            lab_result = pd.DataFrame()

            for lab_object in ['Creatinine', 'Lactate', 'Platelet_count', 'Bilirubin', 'INR']:
            
                lab_measure = lab_object

                current_lab = psql.sqldf(f"SELECT {lab_measure} FROM current_stay")
                current_lab[f'{lab_measure}_up'] = 'N'

                if len(current_lab.dropna())>=2:
                    
                    for idx, measure in enumerate(current_lab.dropna()[f'{lab_measure}'].values):
                        current_idx = current_lab.dropna().index[idx]
                        
                        if idx == 0: #initialize, not calcalurate first for first lab value
                            current_lab.loc[current_idx, f'{lab_measure}_up'] = 0
                            initial_lactate = current_lab.loc[current_idx, f'{lab_measure}']
                        else:
                            if current_lab.loc[current_idx, f'{lab_measure}'] > initial_lactate:
                                current_lab.loc[current_idx, f'{lab_measure}_up'] = 1
                            else:
                                current_lab.loc[current_idx, f'{lab_measure}_up'] = 0
                            initial_lactate = current_lab.loc[current_idx, f'{lab_measure}']
                    
                    
                    # lab up/down forward fill
                    current_lab[f'{lab_measure}_up'] = current_lab[f'{lab_measure}_up'].replace({'N':np.nan})
                    current_lab[f'{lab_measure}_up'] = current_lab[f'{lab_measure}_up'].ffill()
                    
                    # missing lab measurement default value
                    current_lab[f'{lab_measure}'] = current_lab[f'{lab_measure}'].fillna(-100)
                    
                    # lab up/down before first measure
                    current_lab[f'{lab_measure}_up'] = current_lab[f'{lab_measure}_up'].fillna(0)
                    
                else:
                    current_lab[f'{lab_measure}'] = current_lab[f'{lab_measure}'].fillna(-100)
                    current_lab[f'{lab_measure}_up'] = 0
                
                lab_result = pd.concat([lab_result, current_lab[[f'{lab_measure}', f'{lab_measure}_up']]], axis = 1).reset_index(drop=True)
            
            # pharmacuetical phase
            # phar_set = current_stay[['Antibiotics', 'Dobutamine', 'Milrinone', 'Theophyllin', 'Fluid Bolus']]
            # phar_set = phar_set.fillna(0).reset_index(drop=True)
            
            # phar_set['Antibiotics'][phar_set['Antibiotics']>0]=1
            # phar_set['Dobutamine'][phar_set['Dobutamine']>0]=1
            # phar_set['Milrinone'][phar_set['Milrinone']>0]=1
            
            required_columns = ['Antibiotics', 'Dobutamine', 'Milrinone', 'Theophyllin', 'Fluid Bolus']

      
            for v in required_columns:
                if v not in current_stay.columns:
                    current_stay[v] = 0

            phar_set = current_stay[required_columns].fillna(0).reset_index(drop=True)

            phar_set['Antibiotics'] = (phar_set['Antibiotics'] > 0).astype(int)
            phar_set['Dobutamine'] = (phar_set['Dobutamine'] > 0).astype(int)
            phar_set['Milrinone'] = (phar_set['Milrinone'] > 0).astype(int)

            merged_lab_chart = pd.concat([forward_set, lab_result], axis = 1)
            merged_total = pd.concat([merged_lab_chart, phar_set], axis = 1)
            
            result = pd.concat([result, merged_total], axis = 0)
            
            
        dir = f'tabular_records/csv_imputation/part-{parts}.csv'
        result.to_csv(local+dir,index=False)
        print('--')
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")
            
