import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter as tk
import pandas as pd

class SleepStageViewer:
    def __init__(self, fs=128,epoc_dur_in_s=8):
        # サンプル周波数
        self.fs = fs

        #エポックの長さ
        self.epoch_dur = epoc_dur_in_s

        # GUIを作成
        self.root = tk.Tk()
        self.root.title("Sleep Stage Data Viewer")
        self.root.geometry("1200x800")

        # グローバル変数
        self.fig = plt.figure(figsize=(10, 30))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.eeg_data = None
        self.emg_data = None
        self.stage_data = None

        # GUI要素の配置
        self.select_button = tk.Button(self.root, text="Select File", command=self.select_file)
        self.select_button.pack(side=tk.TOP, pady=10)
        self.close_button = tk.Button(self.root, text="Exit", command=self.close_window)
        self.close_button.pack(side=tk.BOTTOM, pady=10)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        

        # ウィンドウを表示
        self.root.mainloop()

    # プロット用の関数
    def plot_data(self):
        # グラフをクリア
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()

        # 脳波データをプロット
        self.ax1.plot(np.arange(len(self.eeg_data)) / self.fs, self.eeg_data)
        self.ax1.set_xlabel("Time [sec]")
        self.ax1.set_ylabel("Amplitude [uV]")
        self.ax1.set_title("EEG Data")

        # 筋電図データをプロット
        self.ax2.plot(np.arange(len(self.emg_data)) / self.fs, self.emg_data)
        self.ax2.set_xlabel("Time [sec]")
        self.ax2.set_ylabel("Amplitude [uV]")
        self.ax2.set_title("EMG Data")

        # ステージデータをプロット
        if self.stage_data is not None:
            self.ax3.plot(np.arange(len(self.stage_data)) / self.fs, self.stage_data)
            self.ax3.set_xlabel("Time [sec]")
            self.ax3.set_ylabel("Sleep Stage")
            self.ax3.set_title("Sleep Stage")

        # グラフを更新
        self.canvas.draw()
    
    def create_widgets(self):
        self.root = tk.Tk()
        self.root.title("Sleep Stage Data Viewer")
        self.root.geometry("800x600")

        select_button = tk.Button(self.root, text="Select File", command=self.select_file)
        select_button.pack(side=tk.TOP, pady=10)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def select_file(self):
        path_eeg = askopenfilename(filetypes=[("EEG Files", "*.pkl")])
        path_emg = path_eeg.replace("EEG.pkl", "EMG.pkl")
        path_stage = askopenfilename(filetypes=[("Sleep stage Files", "*.csv")])

        if path_eeg and path_stage:
            self.load_data(path_eeg, path_stage)
            self.plot_data()

    def load_data(self, path_eeg, path_stage):
        eeg_data_arr = pd.read_pickle(path_eeg)
        self.eeg_data = np.ravel(eeg_data_arr)

        path_emg = path_eeg.replace("EEG.pkl", "EMG.pkl")
        emg_data_arr = pd.read_pickle(path_emg)
        self.emg_data = np.ravel(emg_data_arr)

        df = pd.read_csv(path_stage, comment="#")
        self.stage_data = df.Stage.replace("Wake", 2).replace("NREM", 1).replace("REM", 0)

    def plot_data(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(np.arange(len(self.eeg_data)) / self.fs, self.eeg_data)
        self.ax1.set_xlabel("Time [sec]")
        self.ax1.set_ylabel("Amplitude [uV]")
        self.ax1.set_title("EEG Data")

        self.ax2.plot(np.arange(len(self.emg_data)) / self.fs, self.emg_data)
        self.ax2.set_xlabel("Time [sec]")
        self.ax2.set_ylabel("Amplitude [uV]")
        self.ax2.set_title("EMG Data")

        self.ax3.plot(np.arange(len(self.stage_data))* self.epoch_dur, self.stage_data)
        self.ax3.set_xlabel("Time [sec]")
        self.ax3.set_ylabel("Sleep Stage")
        self.ax3.set_title("Sleep Stage Data")
        self.ax3.set_yticks(np.arange(1, 4))
        self.ax3.set_ylim(-0.5, 2.5)

        self.canvas.draw()

    def close_window(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    viewer = SleepStageViewer()
    viewer.run()











"""
#################

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
"""