import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter as tk
import pandas as pd

# サンプル周波数
fs = 128

# グローバル変数
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
canvas = FigureCanvasTkAgg(fig, master=None)
eeg_data = None
emg_data = None
stage_data = None

# プロット用の関数
def plot_data():
    global canvas, ax1, ax2, ax3, eeg_data, emg_data, stage_data
    
    # グラフをクリア
    ax1.cla()
    ax2.cla()
    ax3.cla()
    
    # 脳波データをプロット
    ax1.plot(np.arange(len(eeg_data)) / fs, eeg_data)
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Amplitude [uV]")
    ax1.set_title("EEG Data")

    # 筋電図データをプロット
    ax2.plot(np.arange(len(emg_data)) / fs, emg_data)
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Amplitude [uV]")
    ax2.set_title("EMG Data")

    # 睡眠ステージデータをプロット
    #ax3.plot(np.arange(len(stage_data)) * 8, stage_data)
    #ax3.set_xlabel("Time [sec]")
    #ax3.set_ylabel("Stage")
    #ax3.set_title("Sleep Stage Data")
    #ax3.set_yticks(np.arange(1, 4))
    #ax3.set_yticklabels(["Wake", "NREM", "REM"])
    #ax3.set_ylim(0.5, 3.5)
    
    # グラフを更新
    canvas.draw()

# ファイル選択用の関数
def select_file():
    global canvas, eeg_data, emg_data, stage_data
    
    # ファイル選択ダイアログを表示
    path_eeg = askopenfilename(filetypes=[("EEG Files", "*.pkl")])
    #path_emg = askopenfilename(filetypes=[("EMG Files", "*.pkl")])
    path_stage = askopenfilename(filetypes=[("Sleep stage Files", "*.csv")])
    
    # ファイルが選択された場合
    if path_eeg:
        # ピックルファイルからデータを読み込む
        eeg_data_arr= pd.read_pickle(path_eeg)
        eeg_data = np.ravel(eeg_data_arr)

        path_emg=path_eeg.replace("EEG.pkl","EMG.pkl")
        emg_data_arr= pd.read_pickle(path_emg)
        emg_data = np.ravel(emg_data_arr)

        # データをプロット
        plot_data()
        
        # グラフをウィンドウに配置
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    if path_stage:    
        df= pd.read_csv(path_stage, comment='#')
        stage_data=df.Stage.replace('Wake', 2).replace('NREM', 1).replace('REM', 0)
        # データをプロット
        plot_data()
        
        # グラフをウィンドウに配置
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# 保存用の関数
def save_file():
    global stage_data
    
    # ファイル保存ダイアログを表示
    file_path = asksaveasfilename(filetypes=[("Text Files", "*.txt")])
    
    # ファイルが選択された場合
    if file_path:
        # データを保存
        np.savetxt(file_path, stage_data)
        
# 睡眠ステージ修正用の関数
def modify_stage():
    global stage_data
    
    # 修正対象の範囲を入力
    start = input("Enter start time (sec): ")
    end = input("Enter end time (sec): ")
    new_stage = input("Enter new stage: ")
    
    # 入力値を数値に変換
    start = int(start)
    end = int(end)
    new_stage = int(new_stage)
    
    # 睡眠ステージを修正
    stage_data[start:end] = new_stage
    
    # データを再プロット
    plot_data()

# GUIを作成
root = tk.Tk()
root.title("Sleep Stage Data Viewer")

# ウィンドウサイズを設定
root.geometry("800x600")

# ファイル選択ボタンを作成
select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(side=tk.TOP, pady=10)

# ウィンドウを表示
root.mainloop()