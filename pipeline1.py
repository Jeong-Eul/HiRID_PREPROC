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

#zip-해제: 데이터가 저장된 파일 내에서 실행하는 것이 좋음

# import tarfile

# tar_path_obs = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/observation_tables_csv.tar.gz'
# output_path_obs ='/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage'

# with tarfile.open(tar_path_obs, 'r:gz') as tar:
#     tar.extractall(path=output_path_obs)
    
    
# tar_path = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/pharma_records_csv.tar.gz'
# output_path ='/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage'

# with tarfile.open(tar_path, 'r:gz') as tar:
#     tar.extractall(path=output_path) 


local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'

obs_col = ['patientid', 'datetime','value', 'variableid']
pharm_col = ['patientid', 'givenat','givendose', 'pharmaid']

def feature_selection(df, item):
    data = df.copy()
    print('Tracking num of observation after featrue selection')
    print('Before selection: ', len(data))
    result = data[data['variableid'].isin(item.item_id.unique())]
    print('After selection: ', len(result))
    
    #feature name mapping
    rename_dict = dict(zip(item.item_id, item.Name))
    result['variableid'] = result['variableid'].map(rename_dict)
    print('Complete featrue selection')
    return result

def tabularization(part_num, valid_stay_ids, selected):
    
    part_csv=pd.DataFrame()   
    
    print('Start Tabularize Part-', part_num)
    print('Number of patient: ', len(selected.patientid.unique()))
    
    for hid in tqdm(valid_stay_ids, desc = f'Tabularize EHR part-{part_num}'):
        gc.collect()

        practice = selected[selected['patientid']==hid]
        val=practice.pivot_table(index='start_time',columns='variableid',values='value').reset_index(drop=True)
        val['patientid'] = hid
        val['Time_since_ICU_admission'] = val.index
        
        if not os.path.exists(local+"tabular_records/csv_tab/"):
            os.makedirs(local+"tabular_records/csv_tab/")
        
        if(part_csv.empty):
            part_csv=val
        else:
            part_csv=pd.concat([part_csv,val],axis=0)
        
        #[ ====== Save temporal data to csv ====== ]
        dir = f'tabular_records/csv_tab/part-{part_num}.csv'
        part_csv.to_csv(local+dir,index=False)
        
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")
    
general = pd.read_csv('/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/general_table.csv')
feature_list = pd.read_csv('/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/feature_summary.csv')

parts_list=[]
for i in range(0, 250):
    parts_list.append(i)

def data_preprocessing_pipeline(part_list, resample_mode):
    
    for parts in part_list:
        print('--')
        observation = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/observation_tables/csv/part-{parts}.csv', usecols=obs_col)
        observation = observation[obs_col]

        pharma = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/pharma_records/csv/part-{parts}.csv', usecols=pharm_col)
        pharma = pharma[pharm_col]

        
        # 변수 id 기준으로 데이터 양 줄이기
        total = pd.concat([observation, pharma.rename(columns={'givenat':'datetime', 'pharmaid':'variableid'}, inplace=True)], axis = 0)
        selected = feature_selection(total, feature_list)

        # 환자 LOS 24시간 이상

        # make ICu intime

        target_patient = general[general['patientid'].isin(selected.patientid.unique())]
        selected_merged=selected.merge(target_patient[['patientid', 'admissiontime']], how='left', left_on='patientid', right_on='patientid')

        selected_merged['datetime']=pd.to_datetime(selected_merged['datetime'])
        selected_merged['admissiontime'] = pd.to_datetime(selected_merged['admissiontime'])
        selected_merged['Time_since_ICU_admission'] = selected_merged['datetime'] - selected_merged['admissiontime']

        del selected_merged['admissiontime']
        del selected_merged['datetime']

        # Resampling

        selected_merged[['start_days', 'dummy','start_hours']] = selected_merged['Time_since_ICU_admission'].astype('str').str.split(' ', -1, expand=True)
        selected_merged[['start_hours','min','sec']] = selected_merged['start_hours'].str.split(':', -1, expand=True)

        resample_mode ='hourly_intervals'

        if resample_mode == '10min_intervals':
            selected_merged['start_time'] = pd.to_numeric(selected_merged['start_days'])*24*60+pd.to_numeric(selected_merged['start_hours'])*60 + (selected_merged['min'].astype('int')//10)*10
        elif resample_mode == 'hourly_intervals':
            selected_merged['start_time'] = pd.to_numeric(selected_merged['start_days'])*24+pd.to_numeric(selected_merged['start_hours'])

        selected_merged=selected_merged.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        selected_merged=selected_merged[selected_merged['start_time']>=0]

        # los >= 24
        los_df = selected_merged.groupby('patientid').max()[['start_time']].reset_index()
        filter_los = los_df[los_df['start_time'] >= 24].patientid.unique()

        filtered_selected = selected_merged[selected_merged['patientid'].isin(filter_los)]


        # Pao2, Fio2, PEEP가 1번이라도 측정되지 않은 stay 제거하기

        filtered_selected=filtered_selected.sort_values(by=['start_time'])

        # Specify the item_ids we are interested in
        required_item_ids = {'PaO2', 'FiO2', 'PEEP'}

        # Find the stay_ids that have all the required item_ids at least once
        valid_stay_ids = filtered_selected[filtered_selected['variableid'].isin(required_item_ids)].groupby('patientid')['variableid'].nunique()
        valid_stay_ids = valid_stay_ids[valid_stay_ids == len(required_item_ids)].index

        selected = filtered_selected[filtered_selected['patientid'].isin(valid_stay_ids)]
        tabularization(parts, valid_stay_ids, selected)
        print('--')

#실행 방법
# data_preprocessing_pipeline(parts_list[0:2], 'hourly_intervals')