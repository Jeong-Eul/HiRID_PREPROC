import argparse
import numpy as np

import pandas as pd
from scipy.stats import zscore
import numpy as np
import warnings
import os
from tqdm import tqdm
import time


pd.set_option('mode.chained_assignment',  None)
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('HiRID processing script', add_help=False)

    # General
    parser.add_argument('--iqr_factor', default=1.5, type=int)
    parser.add_argument('--start_los', default=24, type=int)
    parser.add_argument('--end_los', default=24*7, type=int)
    parser.add_argument('--min_seq_len', default=3, type=int)
    parser.add_argument('--max_seq_len', default=96, type=int)
    parser.add_argument('--prediction_window', default=8, type=int, help='hour')
    
    # Dataset parameters
    parser.add_argument('--observation_path', default='/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/observation_tables/csv/part-', type=str,
                        help='observation(vital, lab) path')
    parser.add_argument('--pharma_path', default='/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/pharma_records/csv/part-', type=str,
                        help='pharmacuetical path')
    parser.add_argument('--general_path', default='/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/general_table.csv', type=str,
                        help='demographic path')
    parser.add_argument('--action', default=False, help='action trend')
    
    # feature information
    parser.add_argument('--obs_feature_path', default='/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/hirid_vital_lab_id.csv', type=str,
                        help='obs_feature_path')
    parser.add_argument('--pharma_feature_path', default='/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/reference_data/hirid_pharma_id.csv', type=str,
                        help='pharma_feature_path')
    
    # save path
    parser.add_argument('--save_pth', default='/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/20251008/', type=str,
                        help='refined dataset save path')
    
    parser.add_argument('--label_save_pth', 
                        default='C:/Users/DAHS/Desktop/Personal_Research/Dataset/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/tabular_records/20251009/', type=str,
                        help='labeled dataset save path')
    
    return parser.parse_args()
    

def feature_selection(df, item):
    data = df.copy()
    result = data[data['variableid'].isin(item.item_id.unique())]
    
    #feature name mapping
    rename_dict = dict(zip(item.item_id, item.Name))
    result['variableid'] = result['variableid'].map(rename_dict)
    return result

def remove_outliers_iqr(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - args.iqr_factor * iqr
    upper = q3 + args.iqr_factor * iqr
    return x.between(lower, upper)

def resampling(df, start_los, end_los):
    df['datetime']=pd.to_datetime(df['datetime'])
    df['admissiontime'] = pd.to_datetime(df['admissiontime'])
    df['Time_since_ICU_admission'] = df['datetime'] - df['admissiontime']

    del df['admissiontime']
    del df['datetime']

    df['Anchor_Time'] = (
        df['Time_since_ICU_admission'].dt.total_seconds() / 60
    ).round(2)

    df=df[df['Anchor_Time']>=0]
    los_df = df.groupby('patientid').max()[['Anchor_Time']].reset_index()
    filter_los = los_df[(los_df['Anchor_Time'] >= start_los*60)&(los_df['Anchor_Time'] <= end_los*60)].patientid.unique()

    filtered_selected = df[df['patientid'].isin(filter_los)]

    filtered_selected=filtered_selected.sort_values(by=['patientid', 'Anchor_Time'])
    
    return filtered_selected

def detect_vasopressor_trend(arr):
    arr = np.array(arr, dtype=np.float32, copy=True)
    arr = np.asarray(arr, dtype=np.float32)
    diff = np.diff(arr)

    state = np.full_like(arr, fill_value='maintain', dtype=object)
    state[np.isnan(arr)] = 'stop'

    valid = ~np.isnan(arr[:-1]) & ~np.isnan(arr[1:])
    state[1:][valid & (diff > 0)] = 'up'
    state[1:][valid & (diff < 0)] = 'down'
    state[1:][valid & (diff == 0)] = 'maintain'

    start_mask = np.isnan(arr[:-1]) & ~np.isnan(arr[1:])
    state[1:][start_mask] = 'up'

    state[0] = 'stop' if np.isnan(arr[0]) else 'up'

    return state

def adm_duration(pharm):
        global refine, inf
        # identify bolus and infusion including start time, end time of administration of pharma

        df = pharm.copy()

        df = df.sort_values('Anchor_Time')
        result = []

        for inf in df.infusionid.unique():
            
            interest = df[df['infusionid']==inf]
            interest['start_time'] = (pd.to_numeric(interest['Anchor_Time'])//5)*5 + 5
            
            # bolus
            if (len(interest)==1) | all(interest.recordstatus.isin([780])):
                interest['recordstatus'] = 780
                result.append(interest)
            
            # infuse
            elif any(interest.recordstatus.isin([524, 776]))&(len(interest)!=1):
                refine = interest[interest.value > 0]
                try:
                    refine.iloc[0, 4] = 524
                except:
                    pass
                
                result.append(refine)
                
            else:
                print(inf)
            
        return pd.concat(result).sort_values(['patientid', 'start_time']).reset_index(drop=True)
    
    
def interpolate_lactate(df, time_col="Time", lactate_col="Lactate"):
    """
    we refered to HiRID paper
    note: these interpolated lactate was not employed during model train/test
    """

    df = df.sort_values(time_col).reset_index(drop=True).copy()
    t = df[time_col].to_numpy()
    l = df[lactate_col].to_numpy()
    n = len(df)
    interp = np.copy(l)

    # 유효 측정 인덱스
    valid_idx = np.where(~np.isnan(l))[0]

    if len(valid_idx) == 0:
        df["Lactate_interp"] = np.nan
        return df
    elif len(valid_idx) == 1:
        # 측정이 1개뿐이면 앞뒤 3시간만 동일값으로 복제
        t0, l0 = t[valid_idx[0]], l[valid_idx[0]]
        interp[:] = np.nan
        interp[(t >= t0 - 180) & (t <= t0 + 180)] = l0
        df["Lactate_interp"] = interp
        return df

    # ---- 구간별 처리 ----
    for i in range(len(valid_idx) - 1):
        i1, i2 = valid_idx[i], valid_idx[i+1]
        t1, t2 = t[i1], t[i2]
        L1, L2 = l[i1], l[i2]
        dt = t2 - t1

        # (1) 상태가 threshold 2 mmol/L을 넘었는지 확인
        
        precede = (L1 >= 2 and L2 < 2)
        follow = (L1 < 2 and L2 >= 2)
        
        cross_2 = follow or precede
        
        if not cross_2:

            # (2) 6시간 이내면 선형보간
            if dt <= 360:
                mask = (t > t1) & (t < t2)
                interp[mask] = L1 + (L2 - L1) * (t[mask] - t1) / (t2 - t1)
            
            # (3) 6시간 초과면 3시간까지만 ffill/bfill
            elif dt > 360:
                mask_f = (t > t1) & (t <= t1 + 180)
                interp[mask_f] = L1
                
                mask_b = (t < t2) & (t >= t2 - 180)
                interp[mask_b] = L2
                
        # (4) lactate 가 2mmol 이상인 경우        
        if cross_2:
            
            if follow: # 두 측정 값 중 후행 값이 넘는 경우: 선행 lactate forward 3시간
                mask_f = (t > t1) & (t <= t1 + 180)
                interp[mask_f] = L1
            else: # 두 측정 값 중 선행 값이 넘는 경우: 후행 lactate forward 3시간
                mask_b = (t < t2) & (t >= t2 - 180)
                interp[mask_b] = L2
                
        first_idx, last_idx = valid_idx[0], valid_idx[-1]
        if l[first_idx] <= 2:  # 정상
            interp[t <= t[first_idx]] = l[first_idx]  # 무제한 backward fill
        else:
            interp[(t >= t[first_idx]) & (t <= t[first_idx] + 180)] = l[first_idx]  # 3시간

        if l[last_idx] <= 2:  # 정상
            interp[t >= t[last_idx]] = l[last_idx]  # 무제한 forward fill
        else:
            interp[(t <= t[last_idx]) & (t >= t[last_idx] - 180)] = l[last_idx]  # 3시간

    df["Lactate_interp"] = interp
    return df

def AnnotationEpisodes(args, part_example):
    
    episodes = []

    for pid, df in part_example.groupby("patientid"):
        one_p = part_example[part_example['patientid']==pid]
        # [['Time', 'MAP', 'Lactate', 'vasopressor']]
        
        one_p['vasopressor_prev'] = one_p['vasopressor'].shift(1).fillna(0)

        df = one_p.sort_values("Time").reset_index(drop=True)
        df = interpolate_lactate(df, time_col="Time", lactate_col="Lactate")
        
        
        labels = []
        for i in range(len(df)):
            t = df.loc[i, "Time"]

            # 45 min window
            window = df[(df["Time"] >= t - 22.5) & (df["Time"] <= t + 22.5)]

            if window.empty:
                labels.append("unknown")
                continue

            # condition
            cond_nonshock = (window["MAP"] > 65) & (window["vasopressor_prev"] == 0) & (window["Lactate_interp"] < 2)
            cond_shock = ((window["MAP"] <= 65) | ((window["vasopressor_prev"] == 1)) & (window["Lactate_interp"] >= 2))

            # duration
            def duration(cond):
                valid_times = window.loc[cond, "Time"].values
                return np.sum(np.diff(valid_times)) if len(valid_times) > 1 else 0

            dur_nonshock = duration(cond_nonshock)
            dur_shock = duration(cond_shock)

            # annotation
            if dur_nonshock >= 30:
                labels.append("non-shock")
            elif dur_shock >= 30:
                labels.append("shock")
            else:
                # unknown: missing or ambiguous
                if window["MAP"].isna().any() or window["Lactate_interp"].isna().any():
                    labels.append("unknown")
                elif (((window["MAP"] <= 65) | (window["vasopressor_prev"] == 1)) & (window["Lactate_interp"] <= 2)).any():
                    labels.append("unknown")
                else:
                    labels.append("unknown")

        df["ShockLabel"] = labels
        df.drop(['vasopressor_prev', 'Lactate_interp'], axis = 1, inplace = True)
        
        
        df = df[df['ShockLabel'] != "unknown"].reset_index(drop=True)
        labels = df['ShockLabel'].values

        start_idx = 0
        ep_idx = 0
        
        for i in range(1, len(labels)):
            prev, curr = labels[i - 1], labels[i]

            # (case A) non-shock -> shock : generate episode
            if prev == "non-shock" and curr == "shock":
                ep_idx += 1
                end_idx = i
                ep = df.iloc[start_idx:end_idx + 1].copy()
                nonshock_count = (ep['ShockLabel'] == 'non-shock').sum()

                # at least 2 non-shock sample (minimum seq)
                if (nonshock_count >= args.min_seq_len - 1) and (len(ep)<=args.max_seq_len): # 2(args.min_seq_len - 1), 60 -> args.min_seq_len, args.max_seq_len
                    ep['episodeid'] = str(ep_idx) + '-' + str(pid)
                    episodes.append(ep)

                # regardless of condition, next start
                start_idx = end_idx + 1

            # (case B) shock -> shock 유지: skip
            elif prev == "shock" and curr == "shock":
                continue

            # (case C) shock -> non-shock: update next start point of episode
            elif prev == "shock" and curr == "non-shock":
                start_idx = i

        # (case D) last non-shock samples
        if start_idx < len(df):
            last_ep = df.iloc[start_idx:].copy()
            nonshock_count = (last_ep['ShockLabel'] == 'non-shock').sum()
            if (nonshock_count >= args.min_seq_len - 1) and (len(ep)<=args.max_seq_len):
                ep_idx += 1
                last_ep['episodeid'] = str(ep_idx) + '-' + str(pid)
                episodes.append(last_ep)
                
    return pd.concat(episodes, axis = 0)


def Labeling(args, parts, labeled_part):
    
    labeled_part['shock_next_8h'] = np.nan
    labeled_part['is_mask'] = 0
    dyn_csv = pd.DataFrame()

    for ep in labeled_part.episodeid.unique():
        
        episode = labeled_part[labeled_part['episodeid']==ep]
        
        if any(episode['ShockLabel'].isin(['shock'])):
            episode['endpoint_window'] = episode['Time'] + args.prediction_window * 60
        
            for idx, row in episode.iterrows():
                current_time = row['Time']
                endpoint_window = row['endpoint_window']

                future_rows = episode[(episode['Time'] > current_time) & (episode['Time'] <= endpoint_window)]

                if any(future_rows['ShockLabel'] == 'shock'):
                    episode.loc[idx, 'shock_next_8h'] = 1
                else:
                    episode.loc[idx, 'shock_next_8h'] = 0
                    
            episode = episode.drop(['endpoint_window'], axis = 1)
        
        else:
            episode['shock_next_8h'] = 0

        if len(episode) < args.max_seq_len:
            pad_length = args.max_seq_len - len(episode)
            pad_rows = pd.DataFrame(0, index = range(pad_length), columns = episode.columns)
            pad_rows['is_mask'] = 1
            pad_rows['patientid'] = episode.patientid.unique()[0]
            pad_rows['episodeid'] = episode.episodeid.unique()[0]
            
            episode = pd.concat([episode, pad_rows], axis = 0)

        else:
            episode = episode.iloc[:args.max_seq_len]
            
        dyn_csv = pd.concat([dyn_csv, episode], axis = 0)
    
        if not os.path.exists(args.label_save_pth):
                os.makedirs(args.label_save_pth)
        file_path = os.path.join(args.label_save_pth, f"part-{parts}.csv")
        dyn_csv.to_csv(file_path,index=False)

    return dyn_csv


def Aggregation(args, parts):
    
    #preparation
    observation = pd.read_csv(args.observation_path+str(parts)+'.csv', usecols=obs_col)
    observation = observation[obs_col]

    pharma = pd.read_csv(args.pharma_path+str(parts)+'.csv', usecols=pharm_col)[pharm_col].rename(columns={'givenat':'datetime',
                                                                                        'pharmaid':'variableid', 
                                                                                        'fluidamount_calc': 'value'})

    general = pd.read_csv(args.general_path)
    general = general[(general['age']>18) & (general['age']<100)]

    feature_obs = pd.read_csv(args.obs_feature_path)
    feature_pha = pd.read_csv(args.pharma_feature_path)
    
    
    # feature selection
    selected_pharma = feature_selection(pharma, feature_pha)
    selected_chart = feature_selection(observation, feature_obs)
    
    
    # outlier removal
    mask = selected_chart.groupby("variableid")["value"].transform(remove_outliers_iqr)

    chart = selected_chart[mask].copy()
    idx = chart[chart['value']==0].index
    chart = chart.drop(index = idx)

    pharm = selected_pharma.copy()
    pharm.loc[pharm['value'] < 0, 'value'] = 0
    
    
    # resampling
    target_patient = general[general['patientid'].isin(chart.patientid.unique())]
    chart=chart.merge(target_patient[['patientid', 'admissiontime']], how='left', left_on='patientid', right_on='patientid')
    pharm=pharm.merge(target_patient[['patientid', 'admissiontime']], how='left', left_on='patientid', right_on='patientid')
    
    pharm = resampling(pharm, args.start_los, args.end_los)
    chart = resampling(chart, args.start_los, args.end_los)
    chart['start_time'] = (pd.to_numeric(chart['Anchor_Time'])//5)*5 + 5

    # sanity
    pharm = pharm[pharm['patientid'].isin(chart.patientid.unique())]
    
    # pharma processing

    interest_code = [524, 8, 520, 780, 776]
    pharm = pharm[pharm['recordstatus'].isin(interest_code)]
    selected_pharm = adm_duration(pharm)
    
    # filtering a target analysis patient
    
    chart.drop(['Time_since_ICU_admission'], axis = 1, inplace=True)
    selected_pharm.drop(['Time_since_ICU_admission'], axis = 1, inplace=True)

    chart['infusionid'] = 'dummy'
    chart['recordstatus'] = 'dummy'

    # Specify the item_ids we are interested in
    required_item_ids = {'HR', 'SBP', 'DBP', 'Lactate'}
    valid_stay_ids = chart[chart['variableid'].isin(required_item_ids)].groupby('patientid')['variableid'].nunique()
    valid_stay_ids = valid_stay_ids[valid_stay_ids == len(required_item_ids)].index

    chart = chart[chart['patientid'].isin(valid_stay_ids)].sort_values(by='Anchor_Time')
    pharm = selected_pharm[selected_pharm['patientid'].isin(valid_stay_ids)].sort_values(by='Anchor_Time')
    
    
    # Causality aware integration
    
    chart_copy = chart.copy()
    pharm_copy = pharm.copy()

    pharm_copy.reset_index(drop=True, inplace=True)
    chart_copy.reset_index(drop=True, inplace=True)

    chart_copy['pivot_time'] = chart_copy['start_time'].copy()
    pharm_copy['pivot_time'] = pharm_copy['start_time'].copy()

    for id in valid_stay_ids:
        
        pharm_ind = pharm_copy[pharm_copy['patientid']==id]
        chart_ind = chart_copy[chart_copy['patientid']==id]

        for bk_id in pharm_ind.start_time.unique():

            ph_bucket = pharm_ind[pharm_ind['start_time']==bk_id]
            ch_bucket = chart_ind[chart_ind['start_time']==bk_id]

            times = sorted(ph_bucket['Anchor_Time'].dropna().unique())

            if len(times) > 1:
                interval = [(0, times[0])]
                interval += [(times[i], times[i+1]) for i in range(len(times)-1)]
            elif len(times) == 1:
                interval = [(np.nan, times[0])]
            else:
                interval = []

            for i, (st, ed) in enumerate(interval):
                
                tr_bucket = ch_bucket[(ch_bucket['Anchor_Time']>=st)&(ch_bucket['Anchor_Time']<ed)]
                    
                idx = tr_bucket.index
                p_idx = ph_bucket.index[i]
                
                if len(tr_bucket)==0:
                    pharm_copy.drop(index = p_idx, inplace = True)
                    pass
                
                chart_copy.loc[idx, 'pivot_time'] = bk_id + i
                pharm_copy.loc[p_idx, 'pivot_time'] = bk_id + i
                
                if i == len(interval)-1:
                    tr_bucket = ch_bucket[(ch_bucket['Anchor_Time']>=ed)]
                    idx = tr_bucket.index
                    
                    chart_copy.loc[idx, 'pivot_time'] = bk_id + i

    pharm_copy.dropna(axis=0, inplace = True) # 삭제한 행에 다시 접근해서 발생한 문제 -> 삭제가 맞음
    
    
    pharm_copy['variableid'] = 'vasopressor'
    column = ['patientid', 'value', 'variableid', 'pivot_time']

    merged = pd.concat([chart_copy[column],pharm_copy[column]], axis = 0)
    
    feat = set(feature_obs.Name.unique()) | set(['vasopressor'])
    part_csv = pd.DataFrame()
    
    for hid in valid_stay_ids:
        
        df2 = merged[merged['patientid'] == hid]
        val = df2.pivot_table(index='pivot_time', columns='variableid', values='value').reset_index()
        val = val.rename(columns={'pivot_time': 'Time'})
        val['patientid'] = hid

        feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
        val=pd.concat([val,feat_df],axis=1)

        # Additional processing

        # 1. vasopressor
        
        if args.action:
            val['vasopressor_vol'] = np.array(val['vasopressor'], dtype=np.float32, copy=True)
            val['action'] = detect_vasopressor_trend(val['vasopressor_vol'].values)
            val['vasopressor_vol'].fillna(0, inplace=True)
        
        val['vasopressor'].fillna(0, inplace=True)
        idx = val[val['vasopressor'] > 0].index
        val.loc[idx, 'vasopressor'] = 1

        # 2. MAP
        val['MAP'] = (val['DBP']*2 + val['SBP'])/3
        
        if part_csv.empty:
            part_csv = val
        else:
            part_csv = pd.concat([part_csv, val], axis=0)

    #[ ====== Save temporal data to csv ====== ]
    if not os.path.exists(args.save_pth):
        os.makedirs(args.save_pth)
    file_path = os.path.join(args.save_pth, f"part-{parts}.csv")
    part_csv.to_csv(file_path,index=False)


if __name__ == '__main__':
    
    # python processing_hirid.py 
    # --observation_path [your path] 
    # --pharma_path [your path] 
    # --general_path [your path] 
    # --obs_feature_path [your path] 
    # --pharma_feature_path [your path]
    # --save_pth [your path]
    # --labeled_save_pth [your path] 
    # --min_seq_len [seq length minimum] 
    # --max_seq_len [seq length maximum] 
    # --prediction_window [prediction window (hour)]
    # --iqr_factor 1.5 
    # --start_los [los min (hour)] 
    # --end_los [los max (hour)]
    
    args = get_args_parser()

    
    obs_col = ['patientid', 'datetime','value', 'variableid']
    pharm_col = ['patientid', 'givenat','fluidamount_calc', 'pharmaid', 'infusionid', 'recordstatus']
    
    print('START: HIRID-PREPROC--')
    
    print('Step 1. : Aggregation')
    start = time.time()
    
    for i in tqdm(range(0, 250)):
        try:
            Aggregation(args, i)
        except:
            pass
        
    print('Step 1 END: time consume:', time.time()-start)
    
    
    print('Step 2. : Annotation & Labeling')
    start = time.time()
    
    for i in tqdm(range(0, 250)):
        try:
            part_example = pd.read_csv(args.save_pth+f"part-{i}.csv")
            annotated_part = AnnotationEpisodes(args, part_example)
            Labeling(args, i, annotated_part)
        except:
            pass
        
    print('Step 2 END: time consume:', time.time()-start)
    
    
    print('Step 3. : Integrate seperated parts into one dataframe (demographic info was included)')
    start = time.time()
    parts = []
    general = pd.read_csv(args.general_path, usecols=['patientid', 'sex', 'age'])
    for i in tqdm(range(0, 250)):
        try:
            part_example = pd.read_csv(args.label_save_pth+f"part-{i}.csv")
            demo = pd.merge(part_example, general, how = 'left', on='patientid')
            
            parts.append(demo)
            
        except:
            pass
        
    result = pd.concat(parts, ignore_index=True)    
    result['sex'] = result['sex'].replace({'M' : 1, 'F':0})
    result.reset_index(drop=True, inplace=True)
    
    result = result.rename(columns = {'sex':'Sex', 'age':'Age'})
    
    idx = result[result['is_mask']==1].index
    result.loc[idx, 'Sex'] = 0
    result.loc[idx, 'Age'] = 0
    
    result.to_csv('HiRID_binary.csv', index=False)
    
    print('======= Summary =======')
    print('Num of obs: ', len(result))
    print('num of patients: ', result.patientid.nunique())
    print('Male patients: ', len(result[result['Sex'] == 1]))
    print('Female patients: ', len(result[result['Sex'] == 0]))
    print('========================')
        
    print('Step 3 END: time consume:', time.time()-start)
    
    print('FINISH: HIRID-PREPROC--')