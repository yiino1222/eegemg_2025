# -*- coding: utf-8 -*-
import pandas as pd
import os
from pathlib import Path
import numpy as np
import pickle
import argparse
import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import textwrap

import faster2lib.eeg_tools as et
import faster2lib.summary_psd as sp
import faster2lib.summary_common as sc
import stage

from datetime import datetime
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter

import warnings
import seaborn as sns
import math

def psd_freq_bins(sample_freq):
    """ assures frequency bins compatibe among different sampling frequencies

    Args:
        sample_freq (int): The sampling frequency

    Returns:
        np.array: An array of frequency bins
    """
    n_fft = int(256 * sample_freq/100)
    # same frequency bins given by signal.welch()
    freq_bins = 1/(n_fft/sample_freq)*np.arange(0, 129)

    return freq_bins

def get_start_indices_of_sleep_stages(stages):
    # Convert the list into a pandas Series
    stages_series = pd.Series(stages)
    # Detect the stages where the stage changes
    change_points = stages_series.ne(stages_series.shift())
    # Return a dictionary where the keys are the start indices and the values are the stages
    return {index: stage for index, stage in stages_series[change_points].items()}

def get_end_indices_of_sleep_stages(stages):
    stages_series = pd.Series(stages)
    change_points = stages_series.ne(stages_series.shift())
    # ★ ここ修正：fillna をやめて shift の fill_value を使う ★
    change_points = change_points.shift(-1, fill_value=False)
    return {index: stage for index, stage in stages_series[change_points].items()}

def get_indices_of_stage(stage_indices, stage_name):
    # Use a dictionary comprehension to extract the indices corresponding to the specified stage
    return [index for index, stage in stage_indices.items() if stage == stage_name]

def get_nrem_spectrum(data, nrem_start_index):
    """
    Returns the frequency spectrum for the NREM start epoch.
    Parameters:
    data (2D array-like): Time-series data of the format [epoch][frequency].
    nrem_start_index (int): The index of the epoch where NREM starts.
    
    Returns:
    1D array-like: The frequency spectrum at the NREM start epoch.
    """
    return data[nrem_start_index]

def calculate_elapsed_time_df(epoch_len_sec, indices):
    # Convert the epoch length from seconds to hours
    epoch_len_hour = epoch_len_sec / 3600

    # Calculate the elapsed time for each NREM index and truncate it to an integer
    elapsed_times = [index * epoch_len_hour for index in indices]
    elapsed_times_floor = [math.floor(time) for time in elapsed_times]

    # Create a dataframe
    df = pd.DataFrame({
        'index': indices,
        'Elapsed_time': elapsed_times,
        'time_in_hour': elapsed_times_floor
    })

    return df

def calculate_avg_spectrum_per_elapsed_time(spectrum_df, elapsed_time_df):
    # Filter the spectrum dataframe to include only the rows corresponding to NREM start epochs
    stage_spectrum_df = spectrum_df.loc[elapsed_time_df['index']]

    # Add the elapsed_time_floor column to the dataframe
    stage_spectrum_df['time_in_hour'] = elapsed_time_df['time_in_hour'].values

    # Group by the elapsed_time_floor column and calculate the mean for each group
    avg_spectrum_df = stage_spectrum_df.groupby('time_in_hour').mean()

    return avg_spectrum_df

def extract_psd_each(psd_info,epoch_len_sec,sample_freq):
    freq_bin=psd_freq_bins(sample_freq)
    
    stage_call=psd_info["stage_call"]
    norm_psd=psd_info["norm"]
    start_idx=get_start_indices_of_sleep_stages(stage_call)
    nrem_start_idx= get_indices_of_stage(start_idx, 'NREM')
    rem_start_idx= get_indices_of_stage(start_idx, 'REM')
    wake_start_idx= get_indices_of_stage(start_idx, 'WAKE')
    end_idx=get_end_indices_of_sleep_stages(stage_call)
    nrem_end_idx= get_indices_of_stage(end_idx, 'NREM')
    rem_end_idx= get_indices_of_stage(end_idx, 'REM')
    wake_end_idx= get_indices_of_stage(end_idx, 'WAKE')
    
    nrem_start_elapsed_time_df = calculate_elapsed_time_df(epoch_len_sec, nrem_start_idx)
    nrem_start_psd_df=calculate_avg_spectrum_per_elapsed_time(pd.DataFrame(norm_psd),
                                                                 nrem_start_elapsed_time_df)   
    nrem_start_psd_df["type"]="nrem_start"
    nrem_end_elapsed_time_df = calculate_elapsed_time_df(epoch_len_sec, nrem_end_idx)
    nrem_end_psd_df=calculate_avg_spectrum_per_elapsed_time(pd.DataFrame(norm_psd),
                                                                 nrem_end_elapsed_time_df)
    nrem_end_psd_df["type"]="nrem_end"
    rem_start_elapsed_time_df = calculate_elapsed_time_df(epoch_len_sec, rem_start_idx)
    rem_start_psd_df=calculate_avg_spectrum_per_elapsed_time(pd.DataFrame(norm_psd),
                                                                 rem_start_elapsed_time_df)   
    rem_start_psd_df["type"]="rem_start"
    rem_end_elapsed_time_df = calculate_elapsed_time_df(epoch_len_sec, rem_end_idx)
    rem_end_psd_df=calculate_avg_spectrum_per_elapsed_time(pd.DataFrame(norm_psd),
                                                                 rem_end_elapsed_time_df)
    rem_end_psd_df["type"]="rem_end"
    
    wake_start_elapsed_time_df = calculate_elapsed_time_df(epoch_len_sec, wake_start_idx)
    wake_start_psd_df=calculate_avg_spectrum_per_elapsed_time(pd.DataFrame(norm_psd),
                                                                 wake_start_elapsed_time_df) 
    wake_start_psd_df["type"]="wake_start"
    wake_end_elapsed_time_df = calculate_elapsed_time_df(epoch_len_sec, wake_end_idx)
    wake_end_psd_df=calculate_avg_spectrum_per_elapsed_time(pd.DataFrame(norm_psd),
                                                                 wake_end_elapsed_time_df)
    wake_end_psd_df["type"]="wake_end"
    
    # デルタ波とシータ波の範囲のカラムを取得
    frequency_columns = [f"f@{i}" for i in freq_bin]
    delta_range=(0,4)
    theta_range=(4,12)
    delta_columns = [col for col in frequency_columns if delta_range[0] <= float(col[2:]) <= delta_range[1]]
    theta_columns = [col for col in frequency_columns if theta_range[0] <= float(col[2:]) <= theta_range[1]]
    df=pd.concat([nrem_start_psd_df,nrem_end_psd_df,rem_start_psd_df,rem_end_psd_df,wake_start_psd_df,wake_end_psd_df])
    df.columns = list(frequency_columns) + list(df.columns[129:])
    #for df in [nrem_start_psd_df,nrem_end_psd_df,rem_start_psd_df,rem_end_psd_df,wake_start_psd_df,wake_end_psd_df]:
        # 各行についてデルタ波とシータ波の平均パワーを計算
    #df['delta_power'] = df[delta_columns].apply(np.mean, axis=1)
    #df['theta_power'] = df[theta_columns].apply(np.mean, axis=1)
    delta_vals = df[delta_columns].mean(axis=1)
    theta_vals = df[theta_columns].mean(axis=1)
    df = df.assign(
        delta_power=delta_vals,
        theta_power=theta_vals,
    )
        
    return df

def extract_psd_from_psdinfo(psd_info_path,epoch_len_sec,sample_freq):
    print(psd_info_path)
    with open(psd_info_path, 'rb') as file:
        # pickle.load()関数でデータを読み込みます。
        psd_info_list = pickle.load(file)
    psd_start_n_end_df_list = []  # 修正: 各データフレームを格納するリストを用意
    
    for psd_info in psd_info_list:
        df_append = extract_psd_each(psd_info, epoch_len_sec, sample_freq).reset_index()
        df_append["exp_label"] = psd_info["exp_label"]
        df_append["mouse_group"] = psd_info["mouse_group"]
        df_append["mouse_ID"] = psd_info["mouse_id"]
        psd_start_n_end_df_list.append(df_append)  # 修正: append()の代わりにリストに追加
    
    # 修正: pd.concat()を使用してリスト内のデータフレームを結合
    psd_start_n_end_df = pd.concat(psd_start_n_end_df_list, ignore_index=True)
    psd_start_n_end_df = psd_start_n_end_df.set_index(["exp_label", "mouse_group", "mouse_ID", "type", "time_in_hour"])
    psd_start_n_end_df = psd_start_n_end_df * 100
    return psd_start_n_end_df


def make_df_from_summary_dic(stats_fname):
    print(stats_fname)
    stats = np.load(stats_fname, allow_pickle=True)[()]
    df_exp_info = stats["stagetime"]
    data_array = stats["stagetime_profile"]
    transition_array = stats["swtrans_profile"]  # [hourly_psw, hourly_pws]
    bout_array = stats["bout_profile"]
    time_offset = stats.get("time_in_hour_offset", 0)
    
    # リストを用意
    stage_merge_list = []
    sw_transition_merge_list = []
    stage_bout_merge_list = []
    
    type_list = ["REM", "NREM", "Wake"]
    
    for i in range(df_exp_info.shape[0]):
        # sleep wake transition
        df_swtansition_append = pd.DataFrame({
            "exp_label": df_exp_info['Experiment label'][i],
            "mouse_group": df_exp_info['Mouse group'][i],
            "mouse_ID": df_exp_info['Mouse ID'][i],
            "hourly_psw": transition_array[i][0],
            "hourly_pws": transition_array[i][1],
            "time_in_hour": np.arange(len(transition_array[i][0])) + time_offset
        })
        sw_transition_merge_list.append(df_swtansition_append)
        
        # bout count and length
        for j, stage in enumerate(type_list):
            if stage == "Wake":
                stage_temp = "WAKE"
            else:
                stage_temp = stage
            bout_array_temp = bout_array[i]
            for hour in range(len(data_array[i][j])):
                filtered_bouts = bout_array_temp[(bout_array_temp.stage == stage_temp) & (bout_array_temp.hour == hour)]
                if filtered_bouts.empty:
                    bout_count = 0
                    mean_duration_sec = 0
                else:
                    bout_count = filtered_bouts.bout_count.iloc[0]
                    mean_duration_sec = filtered_bouts.mean_duration_sec.iloc[0]
                
                stage_bout_append = pd.DataFrame({
                    "exp_label": df_exp_info['Experiment label'][i],
                    "mouse_group": df_exp_info['Mouse group'][i],
                    "mouse_ID": df_exp_info['Mouse ID'][i],
                    "stage": stage,
                    "bout_count": [bout_count],
                    "mean_duration_sec": [mean_duration_sec],
                    "time_in_hour": [hour + time_offset]
                })
                stage_bout_merge_list.append(stage_bout_append)
        
        # hourly stage
        for j, stage in enumerate(type_list):
            df_append = pd.DataFrame({
                "exp_label": df_exp_info['Experiment label'][i],
                "mouse_group": df_exp_info['Mouse group'][i],
                "mouse_ID": df_exp_info['Mouse ID'][i],
                "stage": stage,
                "min_per_hour": data_array[i][j],
                "time_in_hour": np.arange(len(data_array[i][j])) + time_offset
            })
            stage_merge_list.append(df_append)
    
    # 修正: pd.concat()でリスト内のデータフレームを結合
    stage_merge_df = pd.concat(stage_merge_list, ignore_index=True)
    sw_transition_merge_df = pd.concat(sw_transition_merge_list, ignore_index=True)
    stage_bout_merge_df = pd.concat(stage_bout_merge_list, ignore_index=True)
    
    # インデックスを設定
    stage_merge_df = stage_merge_df.set_index(["exp_label", "mouse_group", "mouse_ID", "stage", "time_in_hour"])
    sw_transition_merge_df = sw_transition_merge_df.set_index(["exp_label", "mouse_group", "mouse_ID", "time_in_hour"])
    stage_bout_merge_df = stage_bout_merge_df.set_index(["exp_label", "mouse_group", "mouse_ID", "stage", "time_in_hour"])
    
    return stage_merge_df, sw_transition_merge_df, stage_bout_merge_df

def rename_group_name(merge_df,before_str,after_str):
    index_name_list=list(merge_df.index.names)
    merge_df=merge_df.reset_index()
    merge_df["mouse_group"]=merge_df["mouse_group"].str.replace(before_str,after_str)
    merge_df=merge_df.set_index(index_name_list)
    return merge_df

def rename_group_name_bulk(merge_df, rename_dict=None):
    if not rename_dict:
        return merge_df
    index_name_list = list(merge_df.index.names)
    dfr = merge_df.reset_index()
    for before_str, after_str in rename_dict.items():
        dfr["mouse_group"] = dfr["mouse_group"].str.replace(before_str, after_str, regex=False)
    return dfr.set_index(index_name_list)

def rename_group_of_specified_mouse(merge_df,mouse_id_list,group_str):
    index_name_list=list(merge_df.index.names)
    merge_df=merge_df.reset_index()
    merge_df.loc[merge_df.mouse_ID.isin(mouse_id_list),["mouse_group"]]=group_str
    merge_df=merge_df.set_index(index_name_list)
    return merge_df

def add_index(merge_df,index_name,index_val):
    index_name_list=list(merge_df.index.names)
    #index_name_list.append(index_name)
    #print(index_name_list)
    merge_df=merge_df.reset_index()
    merge_df[index_name]=index_val
    index_name_list.append(index_name)
    merge_df=merge_df.set_index(index_name_list)
    return merge_df

def read_psd_ts_csv(csvpath, stage_type):
    df = pd.read_csv(csvpath, header=[0, 1, 2, 3], index_col=0).reset_index(drop=True)
    merge_list = []  # 修正: 各データフレームを格納するリストを用意
    epoch_len_sec = 8
    for i in range(df.shape[1]):
        array = np.array(df.iloc[:, i])
        bin_num = int(3600 / epoch_len_sec)
        answer = divmod(len(array), bin_num)
        if answer[1] != 0:
            array = array[:-answer[1]]
        array_reshape = array.reshape(-1, bin_num)  # 60 min (3600 sec) bin
        binned_data = np.nanmean(array_reshape, axis=1)
        df_append = pd.DataFrame({
            "exp_label": df.columns[i][0],
            "mouse_group": df.columns[i][1],
            "mouse_ID": df.columns[i][2],
            "stage": stage_type,
            "norm_delta_percentage": binned_data,
            "time_in_hour": np.arange(len(binned_data))
        })
        merge_list.append(df_append)  # 修正: リストに追加

    # 修正: pd.concat()でリスト内のデータフレームを結合
    merge_df = pd.concat(merge_list, ignore_index=True)
    merge_df = merge_df.set_index(["exp_label", "mouse_group", "mouse_ID", "stage", "time_in_hour"])
    return merge_df


def read_hourly_psd_ts_csv(csvpath,stage_type):
    return

def merge_hourly_psd_ts_csv(dir):
    #csv_fname="PSD_norm_hourly_allday_percentage-profile.csv"
    csv_fname="PSD_raw_hourly_allday_percentage-profile.csv"
    freq_bins=sp.psd_freq_bins(sample_freq=128)
    frequency_columns = [f"f@{i}" for i in freq_bins]
    delta_range=(1, 4)
    theta_range=(4, 12)
    # デルタ波とシータ波の範囲のカラムを取得
    delta_columns = [col for col in frequency_columns if delta_range[0] <= float(col[2:]) <= delta_range[1]]
    theta_columns = [col for col in frequency_columns if theta_range[0] <= float(col[2:]) <= theta_range[1]]
    #csv読み込み、カラム名そろえる
    csv_path = os.path.join(dir, csv_fname)
    if not os.path.exists(csv_path):
        fallback_path = Path(dir).parents[1] / "PSD_raw" / csv_fname
        if fallback_path.exists():
            csv_path = str(fallback_path)
        else:
            print(f"[WARN] Missing hourly PSD CSV, skipping: {csv_path}")
            return pd.DataFrame()
    df=pd.read_csv(csv_path).rename(columns={"Experiment label":"exp_label","Mouse group":"mouse_group",
                                                                "Mouse ID":"mouse_ID","Stage":"stage","hour":"time_in_hour"})
    stats_path = Path(dir).parent / "stagetime_stats.npy"
    if stats_path.exists():
        stats = np.load(stats_path, allow_pickle=True)[()]
        time_offset = stats.get("time_in_hour_offset", 0)
        df["time_in_hour"] = df["time_in_hour"] + time_offset
    #nanを前後から補完
    for column in frequency_columns:
        #df[column] = df.groupby(['mouse_ID', 'stage'])[column].apply(lambda group: group.ffill().bfill().fillna(group.mean()))
        df[column] = df.groupby(['mouse_ID', 'stage'])[column].transform(lambda group: group.ffill().bfill().fillna(group.mean()))
    # 各行についてデルタ波とシータ波の平均パワーを計算
    #df['delta_power'] = df[delta_columns].apply(np.mean, axis=1)
    #df['theta_power'] = df[theta_columns].apply(np.mean, axis=1)
    delta_vals = df[delta_columns].mean(axis=1)
    theta_vals = df[theta_columns].mean(axis=1)
    df = df.assign(
        delta_power=delta_vals,
        theta_power=theta_vals,
    )
    df=df.set_index(["exp_label","mouse_group","mouse_ID","stage","time_in_hour"])
    return df

def merge_psd_ts_csv(dir):
    psd_norm_csv_list = [
        "power-timeseries_norm_delta_percentage_NREM.csv",
        "power-timeseries_norm_delta_percentage_Wake.csv",
        "power-timeseries_norm_delta_percentage.csv"
    ]
    stage_type_list = ["NREM", "Wake", "Total"]
    merge_list = []  # 修正: 各データフレームを格納するリストを用意
    
    for i, csv in enumerate(psd_norm_csv_list):
        df_append = read_psd_ts_csv(os.path.join(dir, csv), stage_type_list[i])
        merge_list.append(df_append)  # 修正: リストに追加
    
    # 修正: pd.concat()でリスト内のデータフレームを結合
    merge_df = pd.concat(merge_list, ignore_index=False)  # インデックスを保持する場合は ignore_index=False
    return merge_df

def meta_merge_psd_csv(analyzed_dir_list, subdir_vehicle, subdir_rapalog):
    psd_ts_list = []  # PSD timeseries データフレームを格納するリスト
    psd_profile_list = []  # PSD profile データフレームを格納するリスト

    for dir in analyzed_dir_list:
        # Vehicle データの処理
        df_append_vehicle = merge_hourly_psd_ts_csv(os.path.join(dir, subdir_vehicle, "PSD_raw"))
        if not df_append_vehicle.empty:
            df_append_vehicle = add_index(df_append_vehicle, "drug", "vehicle")
            psd_ts_list.append(df_append_vehicle)  # リストに追加

        # Rapalog データの処理
        df_append_rapalog = merge_hourly_psd_ts_csv(os.path.join(dir, subdir_rapalog, "PSD_raw"))
        if not df_append_rapalog.empty:
            df_append_rapalog = add_index(df_append_rapalog, "drug", "rapalog")
            psd_ts_list.append(df_append_rapalog)  # リストに追加

        # Profile データの処理
        csv_fname = "PSD_norm_allday_percentage-profile.csv"
        vehicle_profile_path = os.path.join(dir, subdir_vehicle, "PSD_norm", csv_fname)
        if not os.path.exists(vehicle_profile_path):
            fallback_vehicle = Path(dir) / "PSD_norm" / csv_fname
            if fallback_vehicle.exists():
                vehicle_profile_path = str(fallback_vehicle)
        if os.path.exists(vehicle_profile_path):
            df_profile_append_vehicle = read_psd_profile_csv(vehicle_profile_path)
            df_profile_append_vehicle = add_index(df_profile_append_vehicle, "drug", "vehicle")
            psd_profile_list.append(df_profile_append_vehicle)  # リストに追加
        else:
            print(f"[WARN] Missing PSD profile CSV, skipping: {vehicle_profile_path}")

        rapalog_profile_path = os.path.join(dir, subdir_rapalog, "PSD_norm", csv_fname)
        if not os.path.exists(rapalog_profile_path):
            fallback_rapalog = Path(dir) / "PSD_norm" / csv_fname
            if fallback_rapalog.exists():
                rapalog_profile_path = str(fallback_rapalog)
        if os.path.exists(rapalog_profile_path):
            df_profile_append_rapalog = read_psd_profile_csv(rapalog_profile_path)
            df_profile_append_rapalog = add_index(df_profile_append_rapalog, "drug", "rapalog")
            psd_profile_list.append(df_profile_append_rapalog)  # リストに追加
        else:
            print(f"[WARN] Missing PSD profile CSV, skipping: {rapalog_profile_path}")

    # リスト内のデータフレームを結合
    merge_psd_ts_df = pd.concat(psd_ts_list, ignore_index=False) if psd_ts_list else pd.DataFrame()
    merge_psd_profile_df = pd.concat(psd_profile_list, ignore_index=False) if psd_profile_list else pd.DataFrame()

    return merge_psd_ts_df, merge_psd_profile_df

def read_psd_profile_csv(csvpath):
    df = pd.read_csv(csvpath)
    exp_label_list = df.iloc[:, 0].unique()
    group_list = df.iloc[:, 1].unique()
    mouse_list = df.iloc[:, 2].unique()
    stage_list = df.iloc[:, 4].unique()
    freq_list = [float(s[2:]) for s in df.columns if s.startswith("f")]
    
    merge_list = []  # 修正: 各データフレームを格納するリストを用意
    
    for l in exp_label_list:
        for g in group_list:
            for m in mouse_list:
                for s in stage_list:
                    try:
                        df_append = pd.DataFrame({
                            "exp_label": l,
                            "mouse_group": g,
                            "mouse_ID": m,
                            "stage": s,
                            "normalized_power": df[
                                (df["Experiment label"] == l) &
                                (df["Mouse group"] == g) &
                                (df["Mouse ID"] == m) &
                                (df["Stage"] == s)
                            ].iloc[0, 6:].values,
                            "frequency": freq_list
                        })
                        merge_list.append(df_append)  # 修正: リストに追加
                    except Exception:
                        pass
    
    # 修正: pd.concat()でリスト内のデータフレームを結合
    merge_df = pd.concat(merge_list, ignore_index=False)
    merge_df = merge_df.set_index(["exp_label", "mouse_group", "mouse_ID", "stage", "frequency"])
    return merge_df


def process_stats_path_list(analyzed_dir_list,vehicle_path,rapalog_path):
    stats_list_vehicle=[]
    stats_list_rapalog=[]
    #vehicle_path="vehicle_60h/stagetime_stats.npy"
    #rapalog_path="rapalog_60h/stagetime_stats.npy"
    #vehicle_path="vehicle_84h_before_24h_after_60h/stagetime_stats.npy"
    #rapalog_path="rapalog_84h_before_24h_after_60h/stagetime_stats.npy"
    for dir in analyzed_dir_list:
        vehicle_stats = os.path.join(dir, vehicle_path)
        rapalog_stats = os.path.join(dir, rapalog_path)
        fallback_stats = os.path.join(dir, "stagetime_stats.npy")
        if not os.path.exists(vehicle_stats) and os.path.exists(fallback_stats):
            vehicle_stats = fallback_stats
        if not os.path.exists(rapalog_stats) and os.path.exists(fallback_stats):
            rapalog_stats = fallback_stats
        stats_list_vehicle.append(vehicle_stats)
        stats_list_rapalog.append(rapalog_stats)
    return stats_list_vehicle,stats_list_rapalog

def process_psd_info_path_list(analyzed_dir_list):
    psd_info_list_vehicle=[]
    psd_info_list_rapalog=[]
    vehicle_path="vehicle_24h_before6h/psd_info_list.pkl"
    rapalog_path="rapalog_24h_before6h/psd_info_list.pkl"
    #vehicle_path="vehicle_84h_before_24h_after_60h/stagetime_stats.npy"
    #rapalog_path="rapalog_84h_before_24h_after_60h/stagetime_stats.npy"
    for dir in analyzed_dir_list:
        vehicle_info = os.path.join(dir, vehicle_path)
        rapalog_info = os.path.join(dir, rapalog_path)
        fallback_info = os.path.join(dir, "psd_info_list.pkl")
        if not os.path.exists(vehicle_info) and os.path.exists(fallback_info):
            vehicle_info = fallback_info
        if not os.path.exists(rapalog_info) and os.path.exists(fallback_info):
            rapalog_info = fallback_info
        psd_info_list_vehicle.append(vehicle_info)
        psd_info_list_rapalog.append(rapalog_info)
    return psd_info_list_vehicle,psd_info_list_rapalog

def merge_individual_df(analyzed_dir_list, vehicle_path, rapalog_path, epoch_len_sec, ample_freq):
    stats_list_vehicle, stats_list_rapalog = process_stats_path_list(analyzed_dir_list, vehicle_path, rapalog_path)
    psd_info_list_vehicle, psd_info_list_rapalog = process_psd_info_path_list(analyzed_dir_list)
    
    meta_merge_list = []  # meta_merge_df用リスト
    meta_merge_list2 = []  # meta_merge_df2用リスト
    meta_merge_list3 = []  # meta_merge_df3用リスト
    psd_start_n_end_list = []  # psd_start_n_end_df用リスト
    
    # Vehicleデータの処理
    for stats in stats_list_vehicle:
        if not os.path.exists(stats):
            print(f"[WARN] Missing stats file, skipping: {stats}")
            continue
        df, df2, df3 = make_df_from_summary_dic(stats)
        df = add_index(df, "drug", "vehicle")
        meta_merge_list.append(df)
        df2 = add_index(df2, "drug", "vehicle")
        meta_merge_list2.append(df2)
        df3 = add_index(df3, "drug", "vehicle")
        meta_merge_list3.append(df3)
    
    for psd_info_list in psd_info_list_vehicle:
        if not os.path.exists(psd_info_list):
            print(f"[WARN] Missing PSD info file, skipping: {psd_info_list}")
            continue
        df4 = extract_psd_from_psdinfo(psd_info_list, epoch_len_sec, ample_freq)
        df4 = add_index(df4, "drug", "vehicle")
        psd_start_n_end_list.append(df4)
    
    # Rapalogデータの処理
    for stats in stats_list_rapalog:
        if not os.path.exists(stats):
            print(f"[WARN] Missing stats file, skipping: {stats}")
            continue
        df, df2, df3 = make_df_from_summary_dic(stats)
        df = add_index(df, "drug", "rapalog")
        meta_merge_list.append(df)
        df2 = add_index(df2, "drug", "rapalog")
        meta_merge_list2.append(df2)
        df3 = add_index(df3, "drug", "rapalog")
        meta_merge_list3.append(df3)
    
    for psd_info_list in psd_info_list_rapalog:
        if not os.path.exists(psd_info_list):
            print(f"[WARN] Missing PSD info file, skipping: {psd_info_list}")
            continue
        df4 = extract_psd_from_psdinfo(psd_info_list, epoch_len_sec, ample_freq)
        df4 = add_index(df4, "drug", "rapalog")
        psd_start_n_end_list.append(df4)
    
    # pd.concatでリスト内のデータフレームを結合
    if not meta_merge_list:
        print("[WARN] No stagetime stats were found for merging; skipping merge.")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, pd.DataFrame()
    meta_merge_df = pd.concat(meta_merge_list, ignore_index=False)
    meta_merge_df2 = pd.concat(meta_merge_list2, ignore_index=False)
    meta_merge_df3 = pd.concat(meta_merge_list3, ignore_index=False)
    psd_start_n_end_df = pd.concat(psd_start_n_end_list, ignore_index=False) if psd_start_n_end_list else pd.DataFrame()
    
    return meta_merge_df, meta_merge_df2, meta_merge_df3, psd_start_n_end_df


def exclude_mouse(meta_merge_df,exclude_mouse_list):
    index_name_list=list(meta_merge_df.index.names)
    meta_merge_df=meta_merge_df.reset_index()
    meta_merge_df=meta_merge_df[~meta_merge_df.mouse_ID.isin(exclude_mouse_list)]
    meta_merge_df=meta_merge_df.set_index(index_name_list)
    return meta_merge_df

def plot_timeseries(ax,x_val,y_val,y_err,plot_color,label):
    ax.plot(x_val,y_val,color=plot_color,label=label)
    ax.fill_between(x_val, y_val-y_err, y_val+y_err, facecolor=plot_color, alpha=0.2)

def calculate_delta(meta_merge_df):
    delta_df=meta_merge_df.loc[pd.IndexSlice[:,:,:,:,:,"rapalog"],:].copy()
    index_name_list=list(delta_df.index.names)
    delta_df=delta_df.reset_index()

    vehicle_df=meta_merge_df.loc[pd.IndexSlice[:,:,:,:,:,"vehicle"],:].copy()
    vehicle_df=vehicle_df.reset_index()
    index_name_list=[s for s in index_name_list if s != 'drug']
    delta_df["rapa-vehicle-delta_min_per_hour"]=delta_df["min_per_hour"]-vehicle_df["min_per_hour"]
    delta_df=delta_df.set_index(index_name_list)
    delta_df.drop(columns=["drug","min_per_hour"],inplace=True)
    return(delta_df)

def merge_sleep_stage_df(analyzed_dir_list,epoch_len_sec,sample_freq):
    vehicle_path="vehicle_24h_before6h/stagetime_stats.npy"
    rapalog_path="rapalog_24h_before6h/stagetime_stats.npy"
    meta_stage_df,meta_merge_df_sw,meta_stage_bout_df,meta_psd_start_end_df=merge_individual_df(analyzed_dir_list,
                                                                          vehicle_path,rapalog_path,epoch_len_sec,sample_freq)
    return meta_stage_df,meta_merge_df_sw,meta_stage_bout_df,meta_psd_start_end_df

def merge_psd_df(analyzed_dir_list):
    subdir_vehicle="vehicle_24h_before6h"
    subdir_rapalog="rapalog_24h_before6h"
    merge_psd_ts_df,merge_psd_profile_df=meta_merge_psd_csv(analyzed_dir_list,subdir_vehicle,subdir_rapalog)
    return merge_psd_ts_df,merge_psd_profile_df

def group_analysis_each_df(df: pd.DataFrame):
    """
    各 DataFrame を
      - keys（mouse_group, drug, stage/type, time_in_hour ...）ごとに
      - 数値カラムだけ mean / sem を計算
      - count は行数カウント（size）で返す
    """

    # どの軸でグループ化するかはこれまでのロジックを踏襲
    if "time_in_hour" in df.index.names:
        if "stage" in df.index.names:
            keys = ["mouse_group", "drug", "stage", "time_in_hour"]
        else:
            keys = ["mouse_group", "drug", "type", "time_in_hour"]
    else:
        if "stage" in df.index.names:
            keys = ["mouse_group", "drug", "stage"]
        else:
            keys = ["mouse_group", "drug", "type"]

    df_grouped = df.groupby(keys)

    # ★ 数値カラムだけを対象にするのが今回の肝 ★
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        raise ValueError(f"数値カラムが見つかりません: columns={df.columns}")

    mean = df_grouped[numeric_cols].mean()
    sem = df_grouped[numeric_cols].sem()

    # count は「何匹いるか」だけわかればいいので size() でOK
    # to_frame() にしておくと今まで通り count.loc[..., 0] で参照できる
    count = df_grouped.size().to_frame()

    return mean, sem, count

def extract_mean_n_err(mean, sem, g_name, drug, sleep_stage, val_name):
    subset = mean.loc[pd.IndexSlice[g_name, drug, sleep_stage, :], val_name]
    subset = subset.sort_index()
    x = subset.index.get_level_values("time_in_hour").to_numpy()
    y = subset.to_numpy()
    err = sem.loc[subset.index, val_name].to_numpy()
    return x, y, err

def extract_mean_n_err_for_PSD(mean,sem,g_name,drug,sleep_stage):
    freq_bins=sp.psd_freq_bins(sample_freq=128)
    frequency_columns = [f"f@{i}" for i in freq_bins]
    y=np.array(mean.loc[pd.IndexSlice[g_name,drug,sleep_stage,:],frequency_columns]).flatten()
    err=np.array(sem.loc[pd.IndexSlice[g_name,drug,sleep_stage,:],frequency_columns]).flatten()
    return y,err

def plot_ts_1group(mean,sem,count,g_name,sleep_stage,ax1,val_name,y_label):
    dark_period=[[0,12],[24,36],[48,60]]
    light_period=[[12,24],[36,48]]
    
    x_val, y, err=extract_mean_n_err(mean,sem,g_name,"vehicle",sleep_stage,val_name)
    sample_n=count.loc[pd.IndexSlice[g_name,"vehicle",sleep_stage,0]][0]
    #label_str="vehicle (n=%d)"%sample_n
    label_str="vehicle"
    plot_timeseries(ax1,x_val,y,err,"k",label_str)

    x_min = float(np.min(x_val)) if x_val.size else 0
    x_max = float(np.max(x_val)) if x_val.size else 0
    x_val, y, err=extract_mean_n_err(mean,sem,g_name,"rapalog",sleep_stage,val_name)
    sample_n=count.loc[pd.IndexSlice[g_name,"rapalog",sleep_stage,0]][0]
    #label_str="rapalog (n=%d)"%sample_n
    label_str="rapalog"
    plot_timeseries(ax1,x_val,y,err,"r",label_str)
    
    if x_val.size:
        x_min = min(x_min, float(np.min(x_val)))
        x_max = max(x_max, float(np.max(x_val)))
    for ax in [ax1]:
        ax.plot([0,60],[0.1,0.1],linewidth=5,color="yellow")
        ax.plot([6.5,17.5],[0.1,0.1],linewidth=5,color="k")
        #ax.plot([37,47],[0.1,0.1],linewidth=10,color="yellow")
        if val_name=="min_per_hour":
            if sleep_stage=="REM":
                ax.set_ylim([0,20])
                ax.set_yticks([0,10,20])
            else:
                ax.set_ylim([0,60])
                ax.set_yticks([0,20,40,60])
        elif val_name=="bout_count":
            if sleep_stage=="REM":
                ax.set_ylim([0,10])
                ax.set_yticks([0,5,10])
            else:
                ax.set_ylim([0,40])
                ax.set_yticks([0,20,40])
        elif val_name=="mean_duration_sec":
            if sleep_stage=="Wake":
                ax.set_ylim([0,2000])
                ax.set_yticks([0,1000,2000])
            elif sleep_stage=="NREM":
                ax.set_ylim([0,600])
                ax.set_yticks([0,300,600])
            elif sleep_stage=="REM":
                ax.set_ylim([0,100])
                ax.set_yticks([0,50,100])
        elif val_name=="norm_delta_percentage":
            ax.set_ylim([0,10])
            ax.set_yticks([0,5,10])
        elif val_name=="delta_power":
            ax.set_ylim([0,20])
            ax.set_yticks([0,10,20])
        elif val_name=="theta_power":
            ax.set_ylim([0,10])
            ax.set_yticks([0,5,10])
        elif val_name=="norm_delta_power":
            ax.set_ylim([0.5,2])
            ax.set_yticks([0.5,1,1.5,2])
        elif val_name=="norm_theta_power":
            ax.set_ylim([0.5,1.5])
            ax.set_yticks([0.5,1,1.5])
        else:
            ax.set_ylim([0,60])
            ax.set_yticks([0,20,40,60])
        #ax.set_ylabel("NREM sleep duration (min/h)")
        ax.set_ylabel(y_label)
        xticks = np.arange(x_min, x_max + 1, 6) if x_max != x_min else [x_min]
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(x) for x in xticks])
        ax.plot([0, 0], [0, ax.get_ylim()[1]], "--", color="gray")
        ax.set_xlabel("Time relative to injection (h)")
        ax.set_xlim([x_min, x_max])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,frameon=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


def plot_ts_mouse_groups(mean, sem, count, mouse_groups, drug, sleep_stage, ax1, val_name, y_label):
    palette = sns.color_palette("colorblind", n_colors=len(mouse_groups))

    plotted_any = False
    x_min = 0
    x_max = 0
    for g_name, color in zip(mouse_groups, palette):
        try:
            x_val, y, err = extract_mean_n_err(mean, sem, g_name, drug, sleep_stage, val_name)
            sample_n = count.loc[pd.IndexSlice[g_name, drug, sleep_stage, 0]][0]
        except KeyError:
            print(f"[WARN] plot_ts_mouse_groups: data missing for group={g_name}, drug={drug}, stage={sleep_stage}")
            continue

        plot_timeseries(ax1, x_val, y, err, color, g_name)
        plotted_any = True
        if x_val.size:
            x_min = min(x_min, float(np.min(x_val)))
            x_max = max(x_max, float(np.max(x_val)))

    if not plotted_any:
        print(f"[WARN] plot_ts_mouse_groups: no data plotted for drug={drug}, stage={sleep_stage}")
        return

    for ax in [ax1]:
        ax.plot([0, 60], [0.1, 0.1], linewidth=5, color="yellow")
        ax.plot([6.5, 17.5], [0.1, 0.1], linewidth=5, color="k")
        if val_name=="min_per_hour":
            if sleep_stage=="REM":
                ax.set_ylim([0,20])
                ax.set_yticks([0,10,20])
            else:
                ax.set_ylim([0,60])
                ax.set_yticks([0,20,40,60])
        elif val_name=="bout_count":
            if sleep_stage=="REM":
                ax.set_ylim([0,10])
                ax.set_yticks([0,5,10])
            else:
                ax.set_ylim([0,40])
                ax.set_yticks([0,20,40])
        elif val_name=="mean_duration_sec":
            if sleep_stage=="Wake":
                ax.set_ylim([0,2000])
                ax.set_yticks([0,1000,2000])
            elif sleep_stage=="NREM":
                ax.set_ylim([0,600])
                ax.set_yticks([0,300,600])
            elif sleep_stage=="REM":
                ax.set_ylim([0,100])
                ax.set_yticks([0,50,100])
        elif val_name=="norm_delta_percentage":
            ax.set_ylim([0,10])
            ax.set_yticks([0,5,10])
        elif val_name=="delta_power":
            ax.set_ylim([0,20])
            ax.set_yticks([0,10,20])
        elif val_name=="theta_power":
            ax.set_ylim([0,10])
            ax.set_yticks([0,5,10])
        elif val_name=="norm_delta_power":
            ax.set_ylim([0.5,2])
            ax.set_yticks([0.5,1,1.5,2])
        elif val_name=="norm_theta_power":
            ax.set_ylim([0.5,1.5])
            ax.set_yticks([0.5,1,1.5])
            ax.set_ylim([0.5,1.5])
        else:
            ax.set_ylim([0,60])
            ax.set_yticks([0,20,40,60])
        ax.set_ylabel(y_label)
        xticks = np.arange(x_min, x_max + 1, 6) if x_max != x_min else [x_min]
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(x) for x in xticks])
        ax.plot([0, 0], [0, ax.get_ylim()[1]], "--", color="gray")
        ax.set_xlabel("Time relative to injection (h)")
        ax.set_xlim([x_min, x_max])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,frameon=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

def plot_PSD_1group(mean,sem,count,g_name,sleep_stage,ax1,y_label):
    freq_bins=sp.psd_freq_bins(sample_freq=128)
    frequency_columns = [f"f@{i}" for i in freq_bins]
    x_val=freq_bins
    
    y,err=extract_mean_n_err_for_PSD(mean,sem,g_name,"vehicle",sleep_stage)
    sample_n=count.loc[pd.IndexSlice[g_name,"vehicle",sleep_stage]].max()
    #label_str="vehicle (n=%d)"%sample_n
    label_str="vehicle"
    plot_timeseries(ax1,x_val,y,err,"k",label_str)

    y,err=extract_mean_n_err_for_PSD(mean,sem,g_name,"rapalog",sleep_stage)
    sample_n=count.loc[pd.IndexSlice[g_name,"rapalog",sleep_stage]].max()
    #label_str="rapalog (n=%d)"%sample_n
    label_str="rapalog"
    plot_timeseries(ax1,x_val,y,err,"r",label_str)
    
    for ax in [ax1]:
        #ax.set_ylabel("NREM sleep duration (min/h)")
        ax.set_ylabel(y_label)
        ax.set_xticks([0,6,12,18,24,30])
        ax.set_xticklabels([0,6,12,18,24,30])
        ax.set_xlim([0,30])
        ax.set_xlabel("EEG Frequency (Hz)")
        #ax.plot([6,6],[0,60],"--",color="gray")
        if y_label=="Norm power change":
            ax.set_yticks([0.5,1,1.5])
            ax.set_yticklabels([0.5,1,1.5])
            ax.set_ylim([0.5,1.5])
        else:
            ax.set_yticks([0,5,10])
            ax.set_yticklabels([0,5,10])
            ax.set_ylim([0,10])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,frameon=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


def plot_PSD_mouse_groups(mean, sem, count, mouse_groups, drug, sleep_stage, ax1, y_label):
    freq_bins=sp.psd_freq_bins(sample_freq=128)
    x_val=freq_bins
    palette = sns.color_palette("colorblind", n_colors=len(mouse_groups))

    plotted_any = False
    for g_name, color in zip(mouse_groups, palette):
        try:
            y,err=extract_mean_n_err_for_PSD(mean,sem,g_name,drug,sleep_stage)
            sample_n=count.loc[pd.IndexSlice[g_name,drug,sleep_stage]].max()
        except KeyError:
            print(f"[WARN] plot_PSD_mouse_groups: data missing for group={g_name}, drug={drug}, stage={sleep_stage}")
            continue
        plot_timeseries(ax1,x_val,y,err,color,g_name)
        plotted_any = True

    if not plotted_any:
        print(f"[WARN] plot_PSD_mouse_groups: no data plotted for drug={drug}, stage={sleep_stage}")
        return

    for ax in [ax1]:
        ax.set_ylabel(y_label)
        ax.set_xticks([0,6,12,18,24,30])
        ax.set_xticklabels([0,6,12,18,24,30])
        ax.set_xlim([0,30])
        ax.set_xlabel("EEG Frequency (Hz)")
        if y_label=="Norm power change":
            ax.set_yticks([0.5,1,1.5])
            ax.set_yticklabels([0.5,1,1.5])
            ax.set_ylim([0.5,1.5])
        else:
            ax.set_yticks([0,5,10])
            ax.set_yticklabels([0,5,10])
            ax.set_ylim([0,10])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,frameon=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


def plot_PSD_1group_zoom(mean,sem,count,g_name,sleep_stage,ax1,y_label):
    freq_bins=sp.psd_freq_bins(sample_freq=128)
    frequency_columns = [f"f@{i}" for i in freq_bins]
    x_val=freq_bins
    
    y,err=extract_mean_n_err_for_PSD(mean,sem,g_name,"vehicle",sleep_stage)
    sample_n=count.loc[pd.IndexSlice[g_name,"vehicle",sleep_stage]].max()
    #label_str="vehicle (n=%d)"%sample_n
    label_str="vehicle"
    plot_timeseries(ax1,x_val,y,err,"k",label_str)

    y,err=extract_mean_n_err_for_PSD(mean,sem,g_name,"rapalog",sleep_stage)
    sample_n=count.loc[pd.IndexSlice[g_name,"rapalog",sleep_stage]].max()
    #label_str="rapalog (n=%d)"%sample_n
    label_str="rapalog"
    plot_timeseries(ax1,x_val,y,err,"r",label_str)
    
    for ax in [ax1]:
        #ax.set_ylabel("NREM sleep duration (min/h)")
        ax.set_ylabel(y_label)
        ax.set_xticks([0,4,8,12])
        ax.set_xticklabels([0,4,8,12])
        ax.set_xlim([0,12])
        ax.set_xlabel("EEG Frequency (Hz)")
        #ax.plot([6,6],[0,60],"--",color="gray")
        if y_label=="Norm power change":
            #ax.set_yticks([0.6,1,1.4])
            #ax.set_yticklabels([0.6,1,1.4])
            #ax.set_ylim([0.6,1.4])
            ax.set_yticks([0.5,1,1.5])
            ax.set_yticklabels([0.5,1,1.5])
            ax.set_ylim([0.5,1.5])
        else:
            ax.set_yticks([0,5,10])
            ax.set_yticklabels([0,5,10])
            ax.set_ylim([0,10])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,frameon=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


def plot_PSD_mouse_groups_zoom(mean, sem, count, mouse_groups, drug, sleep_stage, ax1, y_label):
    freq_bins=sp.psd_freq_bins(sample_freq=128)
    x_val=freq_bins
    palette = sns.color_palette("colorblind", n_colors=len(mouse_groups))

    plotted_any = False
    for g_name, color in zip(mouse_groups, palette):
        try:
            y,err=extract_mean_n_err_for_PSD(mean,sem,g_name,drug,sleep_stage)
            sample_n=count.loc[pd.IndexSlice[g_name,drug,sleep_stage]].max()
        except KeyError:
            print(f"[WARN] plot_PSD_mouse_groups_zoom: data missing for group={g_name}, drug={drug}, stage={sleep_stage}")
            continue
        plot_timeseries(ax1,x_val,y,err,color,g_name)
        plotted_any = True

    if not plotted_any:
        print(f"[WARN] plot_PSD_mouse_groups_zoom: no data plotted for drug={drug}, stage={sleep_stage}")
        return

    for ax in [ax1]:
        ax.set_ylabel(y_label)
        ax.set_xticks([0,4,8,12])
        ax.set_xticklabels([0,4,8,12])
        ax.set_xlim([0,12])
        ax.set_xlabel("EEG Frequency (Hz)")
        if y_label=="Norm power change":
            ax.set_yticks([0.5,1,1.5])
            ax.set_yticklabels([0.5,1,1.5])
            ax.set_ylim([0.5,1.5])
        else:
            ax.set_yticks([0,5,10])
            ax.set_yticklabels([0,5,10])
            ax.set_ylim([0,10])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=10,frameon=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


def plot_bargraph(df, target_group, sleep_stage, y_value, y_label, ax, is_norm=False):
    """
    df: index は何でもOK（MultiIndex / 単一 Index 両方対応）
        必要カラム: mouse_group, mouse_ID, stage, drug, y_value
    """

    # まず列に戻して素直に扱う
    dfr = df.reset_index()

    # 対象グループ & ステージでフィルタ
    sub = dfr[(dfr["mouse_group"] == target_group) &
              (dfr["stage"] == sleep_stage)]

    if sub.empty:
        print(f"[WARN] plot_bargraph: target_group={target_group}, stage={sleep_stage} のデータがありません")
        return

    # barplot 本体（x='drug', y=y_value）
    sns.barplot(
        data=sub,
        x="drug",
        y=y_value,
        hue="drug",              # ← 追加！
        palette={"rapalog": "r", "vehicle": "gray"},
        dodge=False,             # ← hue があってもバーを重ねる
        legend=False,            # ← 凡例を非表示に（警告メッセージが推奨している方法）
        ax=ax
    )

    # rapalog-vehicle のペア線をマウスごとに引く
    for mouse_id, g in sub.groupby("mouse_ID"):
        # 両方揃っているマウスだけ線を引く
        if not {"vehicle", "rapalog"}.issubset(set(g["drug"])):
            continue

        # rapalog, vehicle の値（複数行あっても平均してOK）
        val_rapa = g.loc[g["drug"] == "rapalog", y_value].mean()
        val_veh  = g.loc[g["drug"] == "vehicle", y_value].mean()

        # seaborn の x 軸カテゴリは ['rapalog','vehicle'] の順になる想定なので x=0,1 に線を引く
        ax.plot([0, 1], [val_rapa, val_veh], color="k", alpha=0.7)

    # 以下、元コードの軸スケール設定などはそのまま流用
    for ax in [ax]:
        if y_value == "min_per_hour":
            if sleep_stage == "REM":
                ax.set_ylim([0,10])
                ax.set_yticks([0,5,10])
            else:
                ax.set_ylim([0,65])
                ax.set_yticks([0,30,60])
        elif y_value == "bout_count":
            if sleep_stage == "REM":
                ax.set_ylim([0,10])
                ax.set_yticks([0,5,10])
            else:
                ax.set_ylim([0,40])
                ax.set_yticks([0,20,40])
        elif y_value == "mean_duration_sec":
            if sleep_stage == "Wake":
                ax.set_ylim([0,3000])
                ax.set_yticks([0,1500,3000])
            elif sleep_stage == "NREM":
                ax.set_ylim([0,600])
                ax.set_yticks([0,300,600])
            elif sleep_stage == "REM":
                ax.set_ylim([0,100])
                ax.set_yticks([0,50,100])
        elif y_value == "delta_power":
            if is_norm:
                ax.set_ylim([0.5,2])
                ax.set_yticks([0.5,1,1.5,2.0])
            else:
                ax.set_ylim([0,20])
                ax.set_yticks([0,10,20])
        elif y_value == "theta_power":
            if is_norm:
                ax.set_ylim([0.5,1.5])
                ax.set_yticks([0.5,1,1.5])
            else:
                ax.set_ylim([0,20])
                ax.set_yticks([0,10,20])

        ax.set_ylabel(y_label)
        ax.set_xticks([0,1])
        ax.set_xticklabels(["rapalog","vehicle"], rotation=90)
        ax.set_xlim([-0.5,1.5])
        ax.set_xlabel("")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def plot_bargraph_mouse_groups(df, mouse_groups, drug, sleep_stage, y_value, y_label, ax, is_norm=False):
    dfr = df.reset_index()
    sub = dfr[(dfr["drug"] == drug) & (dfr["stage"] == sleep_stage) & (dfr["mouse_group"].isin(mouse_groups))]

    if sub.empty:
        print(f"[WARN] plot_bargraph_mouse_groups: no data for drug={drug}, stage={sleep_stage}, groups={mouse_groups}")
        return

    palette = dict(zip(mouse_groups, sns.color_palette("colorblind", n_colors=len(mouse_groups))))

    sns.barplot(
        data=sub,
        x="mouse_group",
        y=y_value,
        hue="mouse_group",
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )

    sns.stripplot(
        data=sub,
        x="mouse_group",
        y=y_value,
        hue="mouse_group",
        palette=palette,
        dodge=False,
        jitter=True,
        alpha=0.6,
        size=4,
        legend=False,
        ax=ax,
    )

    for mouse_id, g in sub.groupby("mouse_ID"):
        if len(g["mouse_group"].unique()) < 2:
            continue
        g_sorted = g.sort_values("mouse_group")
        ax.plot([0, 1], g_sorted[y_value].tolist(), color="k", alpha=0.7)

    for ax in [ax]:
        if y_value == "min_per_hour":
            if sleep_stage == "REM":
                ax.set_ylim([0,10])
                ax.set_yticks([0,5,10])
            else:
                ax.set_ylim([0,65])
                ax.set_yticks([0,30,60])
        elif y_value == "bout_count":
            if sleep_stage == "REM":
                ax.set_ylim([0,10])
                ax.set_yticks([0,5,10])
            else:
                ax.set_ylim([0,40])
                ax.set_yticks([0,20,40])
        elif y_value == "mean_duration_sec":
            if sleep_stage == "Wake":
                ax.set_ylim([0,3000])
                ax.set_yticks([0,1500,3000])
            elif sleep_stage == "NREM":
                ax.set_ylim([0,600])
                ax.set_yticks([0,300,600])
            elif sleep_stage == "REM":
                ax.set_ylim([0,100])
                ax.set_yticks([0,50,100])
        elif y_value == "delta_power":
            if is_norm:
                ax.set_ylim([0.5,2])
                ax.set_yticks([0.5,1,1.5,2.0])
            else:
                ax.set_ylim([0,20])
                ax.set_yticks([0,10,20])
        elif y_value == "theta_power":
            if is_norm:
                ax.set_ylim([0.5,1.5])
                ax.set_yticks([0.5,1,1.5])
            else:
                ax.set_ylim([0,20])
                ax.set_yticks([0,10,20])

        ax.set_ylabel(y_label)
        ax.set_xticks(range(len(mouse_groups)))
        ax.set_xticklabels(mouse_groups, rotation=90)
        ax.set_xlim([-0.5, len(mouse_groups) - 0.5])
        ax.set_xlabel("")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

def process_group_analysis(meta_stage_df,meta_merge_df_sw,meta_stage_bout_df,merge_psd_ts_df,merge_psd_profile_df):
    meta_stage_mean,meta_stage_sem,meta_stage_count=group_analysis_each_df(meta_stage_df)
    meta_sw_mean,meta_sw_sem,meta_sw_count=group_analysis_each_df(meta_merge_df_sw)
    meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count=group_analysis_each_df(meta_stage_bout_df)
    meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count=group_analysis_each_df(merge_psd_ts_df)
    meta_psd_profile_mean,meta_psd_profile_sem,meta_psd_profile_count=group_analysis_each_df(merge_psd_profile_df)


def fill_na(df):
    df = df.sort_index()
    # 'mouse_ID', 'stage'に基づいてグループ化し、各グループにおける欠損値を前後のtime_in_hourの値から補完
    df_filled = df.groupby(level=['mouse_ID', 'stage']).apply(lambda group: group.sort_index(level='time_in_hour').bfill().ffill())
    return df_filled

def calculate_mean_power(df: pd.DataFrame, x: int, y: int) -> pd.DataFrame:
    """
    time_in_hour ∈ [x, y] で平均したパワー（f@* と delta/theta_power）をマウス単位へ集約。
    Index: (exp_label は落とす / 他は残す)
    """
    idx_names = list(df.index.names)
    keep_idx = [n for n in idx_names if n not in ("exp_label", "time_in_hour")]

    dfr = df.reset_index()
    if "time_in_hour" not in dfr.columns:
        raise ValueError("time_in_hour が見つかりません。")

    # 周波数列は 'f@' で始まる
    freq_cols = [c for c in dfr.columns if isinstance(c, str) and c.startswith("f@")]
    # 代表的パワー列もあれば平均対象に
    for extra in ["delta_power", "theta_power"]:
        if extra in dfr.columns:
            freq_cols.append(extra)

    dfr = dfr[(dfr["time_in_hour"] >= x) & (dfr["time_in_hour"] <= y)]

    if "stage" in idx_names:
        grp = ["mouse_group", "mouse_ID", "stage", "drug"]
    elif "type" in idx_names:
        grp = ["mouse_group", "mouse_ID", "type", "drug"]
    else:
        grp = ["mouse_group", "mouse_ID", "drug"]

    out = dfr.groupby(grp)[freq_cols].mean().reset_index()
    return out.set_index([c for c in keep_idx if c in out.columns])
def calculate_normalized_psd_ts(df, x, y):
    return 
def calculate_mean_values(meta_stage_df, meta_stage_bout_df, x, y):
    index_name_list = list(meta_stage_df.index.names)
    index_name_list.remove('exp_label')
    index_name_list.remove('time_in_hour')

    meta_stage_df = meta_stage_df.reset_index()
    meta_stage_bout_df = meta_stage_bout_df.reset_index()

    # time window でフィルタ
    meta_stage_df_filtered = meta_stage_df[
        (meta_stage_df['time_in_hour'] >= x) & (meta_stage_df['time_in_hour'] <= y)
    ]
    meta_stage_bout_df_filtered = meta_stage_bout_df[
        (meta_stage_bout_df['time_in_hour'] >= x) & (meta_stage_bout_df['time_in_hour'] <= y)
    ]

    # stage duration の平均（min_per_hour）
    meta_stage_df_grouped = (
        meta_stage_df_filtered
        .groupby(['mouse_group', 'mouse_ID', 'drug', 'stage'])['min_per_hour']
        .mean()
        .reset_index()
    )

    # bout の平均（回数と平均長さ）
    # ★ここを二重カッコにするのがポイント★
    bout_cols = [c for c in ['bout_count', 'mean_duration_sec']
                 if c in meta_stage_bout_df_filtered.columns]

    meta_stage_bout_df_grouped = (
        meta_stage_bout_df_filtered
        .groupby(['mouse_group', 'mouse_ID', 'drug', 'stage'])[bout_cols]
        .mean()
        .reset_index()
    )

    # マージ
    df_merged = pd.merge(
        meta_stage_df_grouped,
        meta_stage_bout_df_grouped,
        on=['mouse_group', 'mouse_ID', 'drug', 'stage'],
        how='inner'
    )

    return df_merged.set_index(index_name_list)

def safe_reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index 名と同名の列があると pandas>=2 系では reset_index で ValueError になる。
    それを避けるため、先に衝突列を改名してから reset する。
    """
    idx_names = [n for n in df.index.names if n is not None]
    overlap = [n for n in idx_names if n in df.columns]
    if overlap:
        # 列側を安全にリネーム（例: stage -> stage_col）
        rename_map = {n: f"{n}_col" for n in overlap}
        df = df.rename(columns=rename_map)
    return df.reset_index()

def calculate_ratio_with_groupby(df: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    """
    MultiIndex をもつ PSD 時系列 DF を、time_in_hour のベースライン区間で正規化（比率×100）する。
    - Index/Columns に 'stage' が両方あっても落ちない（safe_reset_index を使用）
    - groupby キーは実在する列から自動決定
    - time_in_hour は正規化の対象から除外して元の値を維持する
    """

    # 期待するキー候補（順番は粒度の強い順）
    key_candidates = ["exp_label", "mouse_group", "mouse_ID", "stage", "type", "drug"]

    # reset_index の衝突を避けつつ（列の stage と区別）
    df_reset = safe_reset_index(df)

    if "time_in_hour" not in df_reset.columns:
        raise ValueError("time_in_hour が列に見つかりません。Index→列に戻せているか確認してください。")

    # time_in_hour を退避
    original_time_in_hour = df_reset["time_in_hour"].copy()

    # 数値カラム（f@周波数・delta_power・theta_power など）を抽出
    # ★ ここで time_in_hour は除外するのがポイント ★
    numeric_columns = [
        c for c in df_reset.columns
        if pd.api.types.is_numeric_dtype(df_reset[c]) and c != "time_in_hour"
    ]
    if not numeric_columns:
        raise ValueError("正規化する数値カラムが見つかりません。")

    # グループ化キーは「候補のうち df_reset に実在するもの」を使う
    group_keys = [k for k in key_candidates if k in df_reset.columns]

    # ベースライン切り出し
    baseline_df = df_reset[
        (df_reset["time_in_hour"] >= baseline_start) &
        (df_reset["time_in_hour"] <= baseline_end)
    ]
    if baseline_df.empty:
        raise ValueError(f"Baseline 範囲にデータがありません: {baseline_start}–{baseline_end}")

    # ベースライン平均
    baseline_means = (
        baseline_df.groupby(group_keys)[numeric_columns]
        .mean()
        .add_suffix("_baseline")
        .reset_index()
    )

    # マージして正規化
    merged = pd.merge(df_reset, baseline_means, on=group_keys, how="left")
    for col in numeric_columns:
        bcol = f"{col}_baseline"
        merged[col] = (merged[col] / merged[bcol]) 

    # ベースライン列を削除
    merged.drop(columns=[f"{c}_baseline" for c in numeric_columns], inplace=True)

    # time_in_hour を元の値に戻す（ここが重要）
    merged["time_in_hour"] = original_time_in_hour

    # 元と同じ MultiIndex に近い形に戻す
    index_cols = [
        k for k in ["exp_label", "mouse_group", "mouse_ID", "stage", "type", "time_in_hour", "drug"]
        if k in merged.columns
    ]
    result = merged.set_index(index_cols)

    return result

def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    time_in_hour に沿って前後補完。mouse_ID / stage / drug ごとに独立に補完する。
    """
    df = df.sort_index()
    # いまのインデックスから存在するキーだけ拾う
    idx_names = list(df.index.names)
    # time_in_hour を持っている前提
    group_levels = [lvl for lvl in ["mouse_ID", "stage", "type", "drug"] if lvl in idx_names]
    # groupby(level=...) は MultiIndex の level 名を使う
    def _ffill_bfill(g):
        return g.sort_index(level="time_in_hour").bfill().ffill()
    return df.groupby(level=group_levels, group_keys=False).apply(_ffill_bfill)

def merge_n_plot(
    analyzed_dir_list,
    epoch_len_sec,
    sample_freq,
    exclude_mouse_list,
    target_group,
    output_dir,
    group_rename_dic=None,
    comparison_mode="drug",
    comparison_drug="vehicle",
    mouse_groups_to_compare=None,
    quant_time_windows=None,
):
    quant_time_windows = quant_time_windows or {}

    def get_window(key, default):
        window = quant_time_windows.get(key, default)
        if not (isinstance(window, (list, tuple)) and len(window) == 2):
            raise ValueError(f"quant_time_windows['{key}'] should be a list/tuple of two numbers.")
        start, end = window
        return float(start), float(end)

    def format_window_text(window):
        start, end = window
        if start == end:
            return f"Time window: {start:g} h"
        return f"Time window: {start:g}–{end:g} h"

    stage_before_window = get_window("stage_before", (3, 5))
    stage_after_window = get_window("stage_after", (6, 7))
    psd_before_window = get_window("psd_before", (5, 5))
    psd_after_window = get_window("psd_after", (6, 7))
    norm_psd_after_window = get_window("norm_psd_after", psd_after_window)

    stage_after_label = format_window_text(stage_after_window)
    psd_after_label = format_window_text(norm_psd_after_window)

    #merge analyzed data
    meta_stage_df,meta_sw_trans_df,meta_stage_bout_df,meta_psd_start_end_df=merge_sleep_stage_df(analyzed_dir_list,epoch_len_sec,sample_freq)
    if meta_stage_df.empty:
        print("[WARN] No merged stagetime data available; skipping plot generation.")
        return {
            "meta_stage_df": meta_stage_df,
            "meta_sw_trans_df": meta_sw_trans_df,
            "meta_stage_bout_df": meta_stage_bout_df,
            "meta_psd_start_end_df": meta_psd_start_end_df,
        }
    merge_psd_ts_df,merge_psd_profile_df=merge_psd_df(analyzed_dir_list)
    
    #rename group if needed
    meta_stage_df=rename_group_name_bulk(meta_stage_df,group_rename_dic)
    meta_sw_trans_df=rename_group_name_bulk(meta_sw_trans_df,group_rename_dic)
    meta_stage_bout_df=rename_group_name_bulk(meta_stage_bout_df,group_rename_dic)
    meta_psd_start_end_df=rename_group_name_bulk(meta_psd_start_end_df,group_rename_dic)
    merge_psd_ts_df=rename_group_name_bulk(merge_psd_ts_df,group_rename_dic)
    merge_psd_profile_df=rename_group_name_bulk(merge_psd_profile_df,group_rename_dic)

    #exclude mouse if needed
    meta_stage_df=exclude_mouse(meta_stage_df,exclude_mouse_list)
    meta_sw_trans_df=exclude_mouse(meta_sw_trans_df,exclude_mouse_list)
    meta_stage_bout_df=exclude_mouse(meta_stage_bout_df,exclude_mouse_list)
    meta_psd_start_end_df=exclude_mouse(meta_psd_start_end_df,exclude_mouse_list)
    merge_psd_ts_df=exclude_mouse(merge_psd_ts_df,exclude_mouse_list)
    merge_psd_profile_df=exclude_mouse(merge_psd_profile_df,exclude_mouse_list)

    #fill nan
    merge_psd_ts_df=fill_na(merge_psd_ts_df)
      
    # make_normalized_psd_timeseries
    orig_index_names = list(merge_psd_ts_df.index.names)

    # reset_indexを実行
    merge_norm_psd_ts_df=calculate_ratio_with_groupby(merge_psd_ts_df, 0, 5)
        # インデックスレベルの並びだけ、元と揃えられるなら揃える
    if all(name in merge_norm_psd_ts_df.index.names for name in orig_index_names):
        merge_norm_psd_ts_df = (
            merge_norm_psd_ts_df
            .reorder_levels(orig_index_names)
            .sort_index()
        )
    else:
        # 万一名前が完全一致しなくても、とりあえずソートだけ
        merge_norm_psd_ts_df = merge_norm_psd_ts_df.sort_index()
    #merge_norm_psd_ts_df.rename(columns={"delta_power":"norm_delta_power",
    #                                     "theta_power":"norm_theta_power"},inplace=True)

    #quantify timeseries data by time window
    merge_psd_ts_df_before=calculate_mean_power(merge_psd_ts_df, *psd_before_window)
    merge_psd_ts_df_after=calculate_mean_power(merge_psd_ts_df,*psd_after_window)
    #meta_psd_start_end_df_before=calculate_mean_power(meta_psd_start_end_df, 4, 6)
    #meta_psd_start_end_df_after=calculate_mean_power(meta_psd_start_end_df, 7, 9)
    #merge_norm_psd_ts_df_after=calculate_mean_power(merge_norm_psd_ts_df, 7, 9)
    meta_psd_start_end_df_before=calculate_mean_power(meta_psd_start_end_df, *psd_before_window)
    meta_psd_start_end_df_after=calculate_mean_power(meta_psd_start_end_df, *psd_after_window)
    merge_norm_psd_ts_df_after=calculate_mean_power(merge_norm_psd_ts_df, *norm_psd_after_window)
    meta_stage_n_bout_df_before=calculate_mean_values(meta_stage_df, meta_stage_bout_df, *stage_before_window)
    meta_stage_n_bout_df_after=calculate_mean_values(meta_stage_df, meta_stage_bout_df, *stage_after_window)
    
    #rename_column_name
    merge_norm_psd_ts_df.rename(columns={"delta_power":"norm_delta_power",
                                         "theta_power":"norm_theta_power"},inplace=True)
    print(merge_norm_psd_ts_df)
    merge_norm_psd_ts_df.to_csv(os.path.join(output_dir,"merge_norm_psd_ts_df.csv"))

    if comparison_mode not in ("drug", "mouse_group"):
        raise ValueError(f"comparison_mode must be 'drug' or 'mouse_group', got {comparison_mode}")

    if comparison_mode == "mouse_group":
        available_groups = sorted(meta_stage_df.index.get_level_values("mouse_group").unique())
        selected_mouse_groups = mouse_groups_to_compare or available_groups
        if len(selected_mouse_groups) < 2:
            raise ValueError("At least two mouse groups are required when comparison_mode is 'mouse_group'")
    else:
        selected_mouse_groups = [target_group]

    #group analysis of timeseries data
    meta_stage_mean,meta_stage_sem,meta_stage_count=group_analysis_each_df(meta_stage_df)
    #meta_sw_trans_mean,meta_sw_trans_sem,meta_sw_trans_count=group_analysis_each_df(meta_sw_trans_df)
    meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count=group_analysis_each_df(meta_stage_bout_df)
    meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count=group_analysis_each_df(merge_psd_ts_df)
    meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count=group_analysis_each_df(merge_norm_psd_ts_df)
    meta_psd_ts_after_mean,meta_psd_ts_after_sem,meta_psd_ts_after_count=group_analysis_each_df(merge_psd_ts_df_after)
    meta_psd_ts_before_mean,meta_psd_ts_before_sem,meta_psd_ts_before_count=group_analysis_each_df(merge_psd_ts_df_before)
    meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count=group_analysis_each_df(merge_norm_psd_ts_df_after)
    meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count=group_analysis_each_df(meta_psd_start_end_df)
    meta_psd_start_end_df_before_mean,meta_psd_start_end_df_before_sem,meta_psd_start_end_df_before_count=group_analysis_each_df(meta_psd_start_end_df_before)
    meta_psd_start_end_df_after_mean,meta_psd_start_end_df_after_sem,meta_psd_start_end_df_after_count=group_analysis_each_df(meta_psd_start_end_df_after)
        
    meta_stage_df.to_csv(os.path.join(output_dir,"meta_stage_df.csv"))
    meta_sw_trans_df.to_csv(os.path.join(output_dir,"meta_sw_trans_df.csv"))
    meta_stage_bout_df.to_csv(os.path.join(output_dir,"meta_stage_bout_df.csv"))
    merge_psd_ts_df.to_csv(os.path.join(output_dir,"merge_psd_ts_df.csv"))
    merge_psd_profile_df.to_csv(os.path.join(output_dir,"merge_psd_profile_df.csv"))
    meta_psd_start_end_df.to_csv(os.path.join(output_dir,"meta_psd_start_end_df.csv"))
    meta_psd_start_end_df_before.to_csv(os.path.join(output_dir,"meta_psd_start_end_df_before.csv"))
    meta_psd_start_end_df_after.to_csv(os.path.join(output_dir,"meta_psd_start_end_df_after.csv"))
    meta_stage_n_bout_df_before.to_csv(os.path.join(output_dir,"meta_stage_n_bout_df_before.csv"))
    meta_stage_n_bout_df_after.to_csv(os.path.join(output_dir,"meta_stage_n_bout_df_after.csv"))
    merge_psd_ts_df_before.to_csv(os.path.join(output_dir,"merge_psd_ts_df_before.csv"))
    merge_psd_ts_df_after.to_csv(os.path.join(output_dir,"merge_psd_ts_df_after.csv"))
    merge_norm_psd_ts_df_after.to_csv(os.path.join(output_dir,"merge_norm_psd_ts_df_after.csv"))
    meta_norm_psd_ts_mean.to_csv(os.path.join(output_dir,"meta_norm_psd_ts_mean_df.csv"))
    
    print("mouse_group in meta_stage_df:",
      sorted(meta_stage_df.index.get_level_values("mouse_group").unique()))
    print("mouse_group in merge_psd_ts_df:",
        sorted(merge_psd_ts_df.index.get_level_values("mouse_group").unique()))
    print("mouse_group in meta_norm_psd_ts_after_mean:",
        sorted(meta_norm_psd_ts_after_mean.index.get_level_values("mouse_group").unique()))


    def plot_ts_dispatch(mean, sem, count, sleep_stage, ax, val_name, y_label):
        if comparison_mode == "mouse_group":
            plot_ts_mouse_groups(mean, sem, count, selected_mouse_groups, comparison_drug, sleep_stage, ax, val_name, y_label)
        else:
            plot_ts_1group(mean, sem, count, target_group, sleep_stage, ax, val_name, y_label)


    def plot_psd_dispatch(mean, sem, count, sleep_stage, ax, y_label):
        if comparison_mode == "mouse_group":
            plot_PSD_mouse_groups(mean, sem, count, selected_mouse_groups, comparison_drug, sleep_stage, ax, y_label)
        else:
            plot_PSD_1group(mean, sem, count, target_group, sleep_stage, ax, y_label)


    def plot_psd_zoom_dispatch(mean, sem, count, sleep_stage, ax, y_label):
        if comparison_mode == "mouse_group":
            plot_PSD_mouse_groups_zoom(mean, sem, count, selected_mouse_groups, comparison_drug, sleep_stage, ax, y_label)
        else:
            plot_PSD_1group_zoom(mean, sem, count, target_group, sleep_stage, ax, y_label)


    def plot_bar_dispatch(df, sleep_stage, y_value, y_label, ax, is_norm=False):
        if comparison_mode == "mouse_group":
            plot_bargraph_mouse_groups(df, selected_mouse_groups, comparison_drug, sleep_stage, y_value, y_label, ax, is_norm=is_norm)
        else:
            plot_bargraph(df, target_group, sleep_stage, y_value, y_label, ax, is_norm=is_norm)

    
    
    # フォント設定
    plt.rcParams["font.size"] = 16
    plt.rcParams['pdf.fonttype'] = 42

    ##timeserisのプロット
    # gridspecを作成
    row_num=13
    col_num=3
    
    gs = gridspec.GridSpec(row_num,col_num)

    # Figureを作成
    fig = plt.figure(figsize=((col_num*3+1),row_num*4))

    # 各axesを作成
    axes = []
    for row in range(row_num):
        for col in range(col_num):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

    # 1行目: 各ステージの割合の時系列変化
    plot_ts_dispatch(meta_stage_mean,meta_stage_sem,meta_stage_count,
                sleep_stage="Wake",ax=axes[0],val_name="min_per_hour",
                y_label="Wake duration (min/h)")
    plot_ts_dispatch(meta_stage_mean,meta_stage_sem,meta_stage_count,
                sleep_stage="NREM",ax=axes[1],val_name="min_per_hour",
                y_label="NREM sleep duration (min/h)")
    plot_ts_dispatch(meta_stage_mean,meta_stage_sem,meta_stage_count,
                sleep_stage="REM",ax=axes[2],val_name="min_per_hour",
                y_label="REM sleep duration (min/h)")

    # 2行目: 各ステージのブートの回数の時系列変化
    plot_ts_dispatch(meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count,
                sleep_stage="Wake",ax=axes[3],val_name="bout_count",
                y_label="Wake bout (/h)")
    plot_ts_dispatch(meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count,
                sleep_stage="NREM",ax=axes[4],val_name="bout_count",
                y_label="NREM bout (/h)")
    plot_ts_dispatch(meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count,
                sleep_stage="REM",ax=axes[5],val_name="bout_count",
                y_label="REM bout (/h)")

    # 3行目: 各ステージのブートの平均長さの時系列変化
    plot_ts_dispatch(meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count,
                sleep_stage="Wake",ax=axes[6],val_name="mean_duration_sec",
                y_label="mean Wake bout length (s)")
    plot_ts_dispatch(meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count,
                sleep_stage="NREM",ax=axes[7],val_name="mean_duration_sec",
                y_label="mean NREM bout length (s)")
    plot_ts_dispatch(meta_stage_bout_mean,meta_stage_bout_sem,meta_stage_bout_count,
                sleep_stage="REM",ax=axes[8],val_name="mean_duration_sec",
                y_label="mean REM bout length (s)")
    
    # 4行目: 各ステージのデルタパワーの時系列変化
    plot_ts_dispatch(meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count,
                sleep_stage="Wake",ax=axes[9],val_name="delta_power",
                y_label="delta power (%)")
    plot_ts_dispatch(meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count,
                sleep_stage="NREM",ax=axes[10],val_name="delta_power",
                y_label="delta power (%)")
    plot_ts_dispatch(meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count,
                sleep_stage="REM",ax=axes[11],val_name="delta_power",
                y_label="delta power (%)")
    
    # 5行目: 各ステージのデルタパワーの時系列変化
    plot_ts_dispatch(meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count,
                sleep_stage="Wake",ax=axes[12],val_name="norm_delta_power",
                y_label="norm. delta power")
    plot_ts_dispatch(meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count,
                sleep_stage="NREM",ax=axes[13],val_name="norm_delta_power",
                y_label="norm. delta power")
    plot_ts_dispatch(meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count,
                sleep_stage="REM",ax=axes[14],val_name="norm_delta_power",
                y_label="norm. delta power")
    
    # 6行目: 各ステージのシータパワーの時系列変化
    plot_ts_dispatch(meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count,
                sleep_stage="Wake",ax=axes[15],val_name="theta_power",
                y_label="theta power (%)")
    plot_ts_dispatch(meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count,
                sleep_stage="NREM",ax=axes[16],val_name="theta_power",
                y_label="theta power (%)")
    plot_ts_dispatch(meta_psd_ts_mean,meta_psd_ts_sem,meta_psd_ts_count,
                sleep_stage="REM",ax=axes[17],val_name="theta_power",
                y_label="theta power (%)")
    
    # 7行目: 各ステージのデルタパワーの時系列変化
    plot_ts_dispatch(meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count,
                sleep_stage="Wake",ax=axes[18],val_name="norm_theta_power",
                y_label="norm. theta power")
    plot_ts_dispatch(meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count,
                sleep_stage="NREM",ax=axes[19],val_name="norm_theta_power",
                y_label="norm. theta power")
    plot_ts_dispatch(meta_norm_psd_ts_mean,meta_norm_psd_ts_sem,meta_norm_psd_ts_count,
                sleep_stage="REM",ax=axes[20],val_name="norm_theta_power",
                y_label="norm. theta power")

    # 8行目: 薬剤投与前のパワースペクトラム密度
    plot_psd_dispatch(meta_psd_ts_before_mean,meta_psd_ts_before_sem,meta_psd_ts_before_count,
                sleep_stage="Wake",ax=axes[21],y_label="Normalized power (%)")
    plot_psd_dispatch(meta_psd_ts_before_mean,meta_psd_ts_before_sem,meta_psd_ts_before_count,
                sleep_stage="NREM",ax=axes[22],y_label="Normalized power (%)")
    plot_psd_dispatch(meta_psd_ts_before_mean,meta_psd_ts_before_sem,meta_psd_ts_before_count,
                sleep_stage="REM",ax=axes[23],y_label="Normalized power (%)")

    # 9行目: 薬剤投与後のパワースペクトラム密度
    plot_psd_dispatch(meta_psd_ts_after_mean,meta_psd_ts_after_sem,meta_psd_ts_after_count,
                sleep_stage="Wake",ax=axes[24],y_label="Normalized power (%)")
    plot_psd_dispatch(meta_psd_ts_after_mean,meta_psd_ts_after_sem,meta_psd_ts_after_count,
                sleep_stage="NREM",ax=axes[25],y_label="Normalized power (%)")
    plot_psd_dispatch(meta_psd_ts_after_mean,meta_psd_ts_after_sem,meta_psd_ts_after_count,
                sleep_stage="REM",ax=axes[26],y_label="Normalized power (%)")
    
    # 10行目: ブートの最初のエポックのデルタパワーの変化
    """
    plot_ts_1group(meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count,
                target_group,sleep_stage="wake_start",ax1=axes[27],val_name="delta_power",y_label="delta_power")
    plot_ts_1group(meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count,
                target_group,sleep_stage="nrem_start",ax1=axes[28],val_name="delta_power",y_label="delta_power")
    plot_ts_1group(meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count,
                target_group,sleep_stage="rem_start",ax1=axes[29],val_name="delta_power",y_label="delta_power")
    
    # 11行目: ブートの最後のエポックのデルタパワーの変化
    plot_ts_1group(meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count,
                target_group,sleep_stage="wake_end",ax1=axes[30],val_name="delta_power",y_label="delta_power")
    plot_ts_1group(meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count,
                target_group,sleep_stage="nrem_end",ax1=axes[31],val_name="delta_power",y_label="delta_power")
    plot_ts_1group(meta_psd_start_end_df_mean,meta_psd_start_end_df_sem,meta_psd_start_end_df_count,
                target_group,sleep_stage="rem_end",ax1=axes[32],val_name="delta_power",y_label="delta_power")
    """
    # 12行目: 薬剤投与後のパワースペクトラム密度
    plot_psd_dispatch(meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count,
                sleep_stage="Wake",ax=axes[33],y_label="Norm power change")
    plot_psd_dispatch(meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count,
                sleep_stage="NREM",ax=axes[34],y_label="Norm power change")
    plot_psd_dispatch(meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count,
                sleep_stage="REM",ax=axes[35],y_label="Norm power change")
    
    # 13行目: 薬剤投与後のパワースペクトラム密度
    plot_psd_zoom_dispatch(meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count,
                sleep_stage="Wake",ax=axes[36],y_label="Norm power change")
    plot_psd_zoom_dispatch(meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count,
                sleep_stage="NREM",ax=axes[37],y_label="Norm power change")
    plot_psd_zoom_dispatch(meta_norm_psd_ts_after_mean,meta_norm_psd_ts_after_sem,meta_norm_psd_ts_after_count,
                sleep_stage="REM",ax=axes[38],y_label="Norm power change")

    # プロットを表示
    plt.tight_layout()
    plt.show()

    # 図を保存
    fig.savefig(os.path.join(output_dir,"timeseries_and_PSD_plot.pdf"))
            
    ##bargraphのプロット
    
    # gridspecを作成
    # gridspecを作成
    row_num=5
    col_num=3
    
    gs = gridspec.GridSpec(row_num,col_num)

    # Figureを作成
    fig2 = plt.figure(figsize=((col_num*3+1),row_num*4))


    # 各axesを作成
    axes = []
    for row in range(row_num):
        for col in range(col_num):
            ax = fig2.add_subplot(gs[row, col])
            axes.append(ax)
    df=meta_stage_n_bout_df_after

    # 1行目: 各ステージの割合の時系列変化の平均値
    plot_bar_dispatch(df,sleep_stage="Wake",y_value="min_per_hour",
                y_label=f"mean Wake duration (min/h)\n{stage_after_label}",ax=axes[0])
    plot_bar_dispatch(df,sleep_stage="NREM",y_value="min_per_hour",
                y_label=f"mean NREM duration (min/h)\n{stage_after_label}",ax=axes[1])
    plot_bar_dispatch(df,sleep_stage="REM",y_value="min_per_hour",
                y_label=f"mean REM duration (min/h)\n{stage_after_label}",ax=axes[2])

    # 2行目: 各ステージのブート数の時系列変化の平均値
    plot_bar_dispatch(df,sleep_stage="Wake",y_value="bout_count",
                y_label=f"mean Wake bout count (/h)\n{stage_after_label}",ax=axes[3])
    plot_bar_dispatch(df,sleep_stage="NREM",y_value="bout_count",
                y_label=f"mean NREM bout count (/h)\n{stage_after_label}",ax=axes[4])
    plot_bar_dispatch(df,sleep_stage="REM",y_value="bout_count",
                y_label=f"mean REM bout count (/h)\n{stage_after_label}",ax=axes[5])

    # 3行目: 各ステージのブートの長さの時系列変化の平均値
    plot_bar_dispatch(df,sleep_stage="Wake",y_value="mean_duration_sec",
                y_label=f"mean Wake bout length (s)\n{stage_after_label}",ax=axes[6])
    plot_bar_dispatch(df,sleep_stage="NREM",y_value="mean_duration_sec",
                y_label=f"mean NREM bout length (s)\n{stage_after_label}",ax=axes[7])
    plot_bar_dispatch(df,sleep_stage="REM",y_value="mean_duration_sec",
                y_label=f"mean REM bout length (s)\n{stage_after_label}",ax=axes[8])
    
    # 4行目: 各ステージの薬剤投与後のデルタの変化
    df=merge_norm_psd_ts_df_after
    plot_bar_dispatch(df,sleep_stage="Wake",y_value="delta_power",
                y_label=f"relative delta power change\n{psd_after_label}",ax=axes[9],is_norm=True)
    plot_bar_dispatch(df,sleep_stage="NREM",y_value="delta_power",
                y_label=f"relative delta power change\n{psd_after_label}",ax=axes[10],is_norm=True)
    plot_bar_dispatch(df,sleep_stage="REM",y_value="delta_power",
                y_label=f"relative delta power change\n{psd_after_label}",ax=axes[11],is_norm=True)
    
    # 5行目: 各ステージの薬剤投与後のシータの変化
    df=merge_norm_psd_ts_df_after
    plot_bar_dispatch(df,sleep_stage="Wake",y_value="theta_power",
                y_label=f"relative theta power change\n{psd_after_label}",ax=axes[12],is_norm=True)
    plot_bar_dispatch(df,sleep_stage="NREM",y_value="theta_power",
                y_label=f"relative theta power change\n{psd_after_label}",ax=axes[13],is_norm=True)
    plot_bar_dispatch(df,sleep_stage="REM",y_value="theta_power",
                y_label=f"relative theta power change\n{psd_after_label}",ax=axes[14],is_norm=True)


    plt.tight_layout()
    plt.show()
    
    fig2.savefig(os.path.join(output_dir,"bargraph.pdf"))
    
    
def wilcoxon_n_paried_t(stage_df,psd_df,bout_df,target_group,stage):
    print(stage)
    print(target_group)
    print("stage duration")
    data1=stage_df[(stage_df.mouse_group==target_group)&(stage_df.stage==stage)&(stage_df.drug=="vehicle")].min_per_hour
    data2=stage_df[(stage_df.mouse_group==target_group)&(stage_df.stage==stage)&(stage_df.drug=="rapalog")].min_per_hour
    from scipy.stats import wilcoxon
    def safe_wilcoxon(a, b, label):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.size == 0 or b.size == 0:
            print(f"wilcoxon skipped for {label}: empty data")
            return None, None
        if np.allclose(a - b, 0, equal_nan=True):
            print(f"wilcoxon skipped for {label}: all differences are zero")
            return None, None
        return wilcoxon(a, b)
    # ウィルコクソンの符号順位検定
    statistic, p_value = safe_wilcoxon(data1, data2, "stage duration")
    print("wilcoxon")
    print('Statistic:', statistic)
    print('p-value:', p_value)

    from scipy.stats import ttest_rel
    statistic, p_value = ttest_rel(data1, data2)
    print("paired t test")
    print('Statistic:', statistic)
    print('p-value:', p_value)
    
    print("stage bout count")
    data1=bout_df[(bout_df.mouse_group==target_group)&(bout_df.stage==stage)&(bout_df.drug=="vehicle")].bout_count
    data2=bout_df[(bout_df.mouse_group==target_group)&(bout_df.stage==stage)&(bout_df.drug=="rapalog")].bout_count
    
    # ウィルコクソンの符号順位検定
    statistic, p_value = safe_wilcoxon(data1, data2, "stage bout count")
    print("wilcoxon")
    print('Statistic:', statistic)
    print('p-value:', p_value)
    
    print("stage bout length")
    data1=bout_df[(bout_df.mouse_group==target_group)&(bout_df.stage==stage)&(bout_df.drug=="vehicle")].mean_duration_sec
    data2=bout_df[(bout_df.mouse_group==target_group)&(bout_df.stage==stage)&(bout_df.drug=="rapalog")].mean_duration_sec

    # ウィルコクソンの符号順位検定
    statistic, p_value = safe_wilcoxon(data1, data2, "stage bout length")
    print("wilcoxon")
    print('Statistic:', statistic)
    print('p-value:', p_value)

    print("norm delta power")
    data1=psd_df[(psd_df.mouse_group==target_group)&(psd_df.stage==stage)&(psd_df.drug=="vehicle")].delta_power
    data2=psd_df[(psd_df.mouse_group==target_group)&(psd_df.stage==stage)&(psd_df.drug=="rapalog")].delta_power

    # ウィルコクソンの符号順位検定
    statistic, p_value = safe_wilcoxon(data1, data2, "norm delta power")
    print("wilcoxon")
    print('Statistic:', statistic)
    print('p-value:', p_value)

    # paired t test
    statistic, p_value = ttest_rel(data1, data2)
    print("paired t test")
    print('Statistic:', statistic)
    print('p-value:', p_value)

    
    print("norm theta power")
    data1=psd_df[(psd_df.mouse_group==target_group)&(psd_df.stage==stage)&(psd_df.drug=="vehicle")].theta_power
    data2=psd_df[(psd_df.mouse_group==target_group)&(psd_df.stage==stage)&(psd_df.drug=="rapalog")].theta_power

    # ウィルコクソンの符号順位検定
    statistic, p_value = safe_wilcoxon(data1, data2, "norm theta power")
    print("wilcoxon")
    print('Statistic:', statistic)
    print('p-value:', p_value)

    # paired t test
    statistic, p_value = ttest_rel(data1, data2)
    print("paired t test")
    print('Statistic:', statistic)
    print('p-value:', p_value)


    return
