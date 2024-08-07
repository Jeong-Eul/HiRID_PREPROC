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

if not os.path.exists(local+"tabular_records/csv_labeling_10/"):
            os.makedirs(local+"tabular_records/csv_labeling_10/")

def Labeling_ARDS(part_list):
    print("[ EXCUTING DATA Labeling ]")
    for parts in part_list:
        print(f'Part {parts} processing......')

        part = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_imputation/part-{parts}.csv')

        part['Annotation'] = np.nan
        
        for stay in tqdm(part.patientid.unique()):

            interest = part[part['patientid']==stay]
            interest = interest.reset_index()

            available = psql.sqldf("SELECT * FROM interest WHERE FiO2 != -100 and PaO2 != -100 and PEEP IS NOT NULL", locals())
            available = available.set_index('index', drop=True)

            if len(available)>0:
                
                available['PaO2/FiO2'] = 100*(available['PaO2'] / (available['FiO2'] + 0.000000001))
                idx_ards = available[(available['PEEP'] >= 5.0) & (available['PaO2/FiO2'] <=300)].index
                idx_no_ards = available.index.difference(idx_ards)
                all_idx = available.index
                
                part.loc[idx_ards, 'Annotation'] = 'ARDS'
                part.loc[idx_no_ards, 'Annotation'] = 'Not ARDS'
                part.loc[all_idx, 'PaO2/FiO2'] = available['PaO2/FiO2'].copy()
                part['PaO2/FiO2'] = part['PaO2/FiO2'].fillna(-100)
                
            else:
                idx = interest.index
                part.loc[idx, 'Annotation'] = 'Not ARDS'
                part.loc[idx, 'PaO2/FiO2'] = -100
            
                
        part['Annotation'] = part['Annotation'].fillna('Not ARDS')        

        part['ARDS_next_12h'] = np.nan
        

        for stay_id in tqdm(part['patientid'].unique()):
            stay_df = part[part['patientid'] == stay_id].sort_values(by='Time_since_ICU_admission')
            stay_df['endpoint_window'] = stay_df['Time_since_ICU_admission'] + 12

            for idx, row in stay_df.iterrows():
                current_time = row['Time_since_ICU_admission']
                endpoint_window = row['endpoint_window']

                future_rows = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]

                if any(future_rows['Annotation'] == 'ARDS'):
                    part.loc[idx, 'ARDS_next_12h'] = 1
                else:
                    part.loc[idx, 'ARDS_next_12h'] = 0
                    
        dir = f'tabular_records/csv_labeling/part-{parts}.csv'
        part.to_csv(local+dir,index=False)
        print('--')
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")
    
    
def Labeling_SHOCK(part_list):
    print("[ EXCUTING DATA Labeling ]")
    for parts in part_list:
        print(f'Part {parts} processing......')

        part = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_imputation_10/part-{parts}.csv')

        part['Annotation'] = np.nan
        
        for stay in tqdm(part.patientid.unique()):
            interest = part[part['patientid'] == stay]

            available = interest.dropna(subset=['MAP', 'vasopressor'])

            if len(available) > 0:
                
                condition1 = available['vasopressor'] > 0
                condition2 = (available['MAP'] <= 65.0) | (available['vasopressor'] > 0)
                condition3 = available['Lactate'] >= 2
                
                idx_shock_vaso = available[condition1].index
                idx_shock_map_lactate = available[(condition2)&condition3].index
                
                idx_shock = idx_shock_vaso.union(idx_shock_map_lactate)
                
                idx_no_shock = available.index.difference(idx_shock)

                part.loc[idx_shock, 'Annotation'] = 'shock'
                part.loc[idx_no_shock, 'Annotation'] = 'non-shock'
                
            else:
                part.loc[interest.index, 'Annotation'] = 'non-shock'
                
                    
            part['Annotation'] = part['Annotation'].fillna('non-shock')        

            part['shock_next_6h'] = np.nan
        

        for stay_id in tqdm(part['patientid'].unique()):
            stay_df = part[part['patientid'] == stay_id].sort_values(by='Time_since_ICU_admission')
            stay_df['endpoint_window'] = stay_df['Time_since_ICU_admission'] + 36

            for idx, row in stay_df.iterrows():
                current_time = row['Time_since_ICU_admission']
                endpoint_window = row['endpoint_window']

                future_rows = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]

                if any(future_rows['Annotation'] == 'shock'):
                    part.loc[idx, 'shock_next_6h'] = 1
                else:
                    part.loc[idx, 'shock_next_6h'] = 0
                    
        dir = f'tabular_records/csv_labeling_10/part-{parts}.csv'
        part.to_csv(local+dir,index=False)
        print('--')
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")