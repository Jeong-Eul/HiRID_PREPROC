import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
import warnings
pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning)
local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'

# col = ['Lactate','Milrinone','MAP','Fluid Bolus','Platelet count','ABPs','Bilirubin','FiO2','SpO2',
#       'Time_since_ICU_admission','Dobutamine','Temperature','PEEP','ABPd','Urine_output','Theophyllin',
#       'Antibiotics','INR','HR','pH','Creatinine','patientid','PaO2','Respiratory_rate']

if not os.path.exists(local+"tabular_records/csv_exclusion_10/"):
            os.makedirs(local+"tabular_records/csv_exclusion_10/")

def Exclusion_Checker(part_list):
    print("[ EXCUTING DATA Labeling ]")
    
    logs = []

    total_pre_removal_count = 0
    total_post_removal_count = 0

    for parts in part_list:
        print(f'Part {parts} processing......')

        part = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_labeling_10/part-{parts}.csv')
        
        pre_removal_count = part.patientid.nunique()
        total_pre_removal_count += pre_removal_count
        
        split_data = []
        current_part = []
        
        for stay in tqdm(part.patientid.unique()):
            
            event_occurred = False
            interest = part[part['patientid'] == stay]

            if any(interest.head(36)['Annotation'] == 'shock'): # ICU 입원 후 6시간을 보내기 전에 shock이 온 경우
                continue
                
            elif len(interest) < 36 : # ICU LOS가 6시간보다 적은 경우
                continue
                
            else:
                for _, row in interest.iterrows():

                    if event_occurred:
                        event_occurred = False
                        break
                    else:
                        current_part.append(row)
                        if row['Annotation'] == 'shock':
                            split_data.append(pd.DataFrame(current_part))
                            event_occurred = True
                            current_part = []
                
   
        if len(split_data) > 0:
        
            new_part = pd.concat(split_data).reset_index(drop=True)      
            post_removal_count = new_part.patientid.nunique()
            total_post_removal_count += post_removal_count
            
            dir = f'tabular_records/csv_exclusion_10/part-{parts}.csv'
            new_part.to_csv(local+dir,index=False)


            logs.append((parts, pre_removal_count, post_removal_count))
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")
    
    print("\n[ TOTAL SUMMARY ]")
    print(f'전체 파트: 제거 전 {total_pre_removal_count}명, 제거 후 {total_post_removal_count}명')