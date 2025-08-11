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

general = pd.read_csv('/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/general_table.csv')
feature_list = pd.read_csv('/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/feature_summary_causal.csv')

def remove_outliers_by_percentile(df, lower=2, upper=98):
    df_cleaned = df.copy()
    df = df.iloc[:, ~df.columns.isin(['Time_since_ICU_admission', 'patientid', 'Treatment-Bolus', 'Treatment-IV'])]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_bound = df[col].quantile(lower / 100)
            upper_bound = df[col].quantile(upper / 100)

            # 하위 2%, 상위 98% 바깥의 값은 NaN 처리
            df_cleaned[col] = df[col].where((df[col] >= lower_bound) & (df[col] <= upper_bound), np.nan)

    return df_cleaned



local = '/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'

obs_col = ['patientid', 'datetime','value', 'variableid']
pharm_col = ['patientid', 'givenat','givendose', 'pharmaid', 'infusionid', 'recordstatus']

def feature_selection(df, item):
    data = df.copy()
    # print('Tracking num of observation after featrue selection')
    # print('Before selection: ', len(data))
    result = data[data['variableid'].isin(item.item_id.unique())]
    # print('After selection: ', len(result))
    
    #feature name mapping
    rename_dict = dict(zip(item.item_id, item.Name))
    result['variableid'] = result['variableid'].map(rename_dict)
    # print('Complete featrue selection')
    return result

def tabularization(part_num, valid_stay_ids, selected, resample_mode):
    
    feat = feature_list.Name.unique()
    
    feat = list(set(feat) - set(['Norephinephrine', 'Adrenalin', 'Glypressin', 'Vasopressin', 'Inotropic']) | set(['Treatment-Bolus', 'Treatment-IV']))
    
    part_csv=pd.DataFrame()   
    # print('Start Tabularize Part-', part_num)
    # print('Number of patient: ', len(selected.patientid.unique()))
    local = f"/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/csv_tab_{resample_mode}/"
    for hid in valid_stay_ids:
        gc.collect()
        practice = selected[selected['patientid'] == hid]
        val = practice.pivot_table(index='start_time', columns='variableid', values='value').reset_index()
        val = val.rename(columns={'start_time': 'Time_since_ICU_admission'})
        val['patientid'] = hid

        if part_csv.empty:
            part_csv = val
        else:
            part_csv = pd.concat([part_csv, val], axis=0)
        
        part_csv = remove_outliers_by_percentile(part_csv, lower=2, upper=98)
        feat_df=pd.DataFrame(columns=list(set(feat)-set(part_csv.columns)))
        part_csv=pd.concat([part_csv,feat_df],axis=1)
        #[ ====== Save temporal data to csv ====== ]
        if not os.path.exists(local):
            os.makedirs(local)
        file_path = os.path.join(local, f"part-{part_num}.csv")
        part_csv.to_csv(file_path,index=False)
        
    # print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")
    

parts_list=[]
for i in range(0, 250):
    parts_list.append(i)

def data_preprocessing_pipeline(part_list, resample_mode, start_los, end_los):
    print(f'Resampling mode: {resample_mode}')
    print('Start data preprocessing..')
    for parts in tqdm(part_list, desc = 'Tabularize EHR'):
        # print('--')
        observation = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/observation_tables/csv/part-{parts}.csv', usecols=obs_col)
        observation = observation[obs_col]

        pharma = pd.read_csv(f'/Users/DAHS/Desktop/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/pharma_records/csv/part-{parts}.csv', usecols=pharm_col)
        pharma = pharma[pharm_col]

        new_pharma = pharma.rename(columns={'givenat':'datetime', 'pharmaid':'variableid', 'givendose': 'value'})
        
        # 변수 id 기준으로 데이터 양 줄이기
        
        pharma = pharma[pharma['pharmaid']!=410] # noise
        new_pharma = pharma.rename(columns={'givenat':'datetime', 'pharmaid':'variableid', 'givendose': 'value'})

        selected_pharma = feature_selection(new_pharma, feature_list)
        selected_chart = feature_selection(observation, feature_list)

        interest_code = [524, 8, 520, 524, 780]
        infusion_code = [524, 8, 520, 524]
        bolus_code = [780]

        ph = selected_pharma[selected_pharma['recordstatus'].isin(interest_code)]

        infusion_df_list = []

        for infusion in ph.infusionid.unique():
            infusion_df = ph.loc[ph.infusionid == infusion, :].copy()
            
            if infusion_df.value.sum() == 0:
                pass
            
            else:
                # single infusion의 기록이 1개인데, Bolus로 표시 안된 경우 재정제
                if len(infusion_df) == 1 and infusion_df.iloc[0].recordstatus != 780:
                    if infusion_df['value'].iloc[0] == 0:
                        pass
                    else:
                        infusion_df.loc[infusion_df.index[0], 'recordstatus'] = 780
                    
            infusion_df_list.append(infusion_df)
            
        processed_ph = pd.concat(infusion_df_list).drop(['infusionid'], axis = 1)
        processed_ph.loc[processed_ph['recordstatus'].isin(infusion_code), 'recordstatus'] = 'IV'
        processed_ph.loc[processed_ph['recordstatus'].isin(bolus_code), 'recordstatus'] = 'Bolus'

        processed_ph['variableid'] = 'Treatment'
        processed_ph['variableid'] = processed_ph['variableid']+'-'+processed_ph['recordstatus']


        selected = pd.concat([selected_chart, processed_ph.drop(['recordstatus'], axis = 1)], axis = 0).reset_index(drop=True)

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

        if resample_mode == '10min_intervals':
            selected_merged['start_time'] = pd.to_numeric(selected_merged['start_days'])*24*60 + pd.to_numeric(selected_merged['start_hours'])*60 + (selected_merged['min'].astype('int')//10)*10
            adj_coeff = 6
        elif resample_mode == '5min_intervals':
            selected_merged['start_time'] = ((pd.to_numeric(selected_merged['start_days'])*24*60 + 
                                pd.to_numeric(selected_merged['start_hours'])*60 + 
                                (selected_merged['min'].astype('int')))//5)*5
            adj_coeff = 60
        elif resample_mode == 'hourly_intervals':
            selected_merged['start_time'] = pd.to_numeric(selected_merged['start_days'])*24+pd.to_numeric(selected_merged['start_hours'])
            adj_coeff = 1
        
        
        selected_merged=selected_merged.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        selected_merged=selected_merged[selected_merged['start_time']>=0]

        # los >= 24
        los_df = selected_merged.groupby('patientid').max()[['start_time']].reset_index()
        filter_los = los_df[(los_df['start_time'] >= start_los*adj_coeff)&(los_df['start_time'] <= end_los*adj_coeff)].patientid.unique()

        filtered_selected = selected_merged[selected_merged['patientid'].isin(filter_los)]


        # ABPd, ABPs, Lactate 1번이라도 측정되지 않은 stay 제거하기

        filtered_selected=filtered_selected.sort_values(by=['start_time'])

        # Specify the item_ids we are interested in
        required_item_ids = {'HR', 'ABPd', 'ABPs'}
        
        patient_varids = (
                            filtered_selected
                            .groupby('patientid')['variableid']
                            .apply(set)
                        )
        
        def has_required_vars(varids):
            has_basic = required_item_ids.issubset(varids)
            has_treatment = any(('Treatment-IV' in v or 'Treatment-Bolus' in v) for v in patient_varids)
            return has_basic and has_treatment

        # Find the stay_ids that have all the required item_ids at least once
        # valid_stay_ids = filtered_selected[filtered_selected['variableid'].isin(required_item_ids)].groupby('patientid')['variableid'].nunique()
        # valid_stay_ids = valid_stay_ids[valid_stay_ids == len(required_item_ids)].index

        valid_stay_ids = patient_varids[patient_varids.apply(has_required_vars)].index
        
        selected = filtered_selected[filtered_selected['patientid'].isin(valid_stay_ids)]
        
        if len(valid_stay_ids)==0:
            pass
        else:
            tabularization(parts, valid_stay_ids, selected, resample_mode)
        # print('--')
    print('FINISH')
#실행 방법
# data_preprocessing_pipeline(parts_list[0:2], 'hourly_intervals')