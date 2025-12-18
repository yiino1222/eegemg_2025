import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter as tk

# グローバル変数
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=None)
data = None

# プロット用の関数
def plot_data():
    global ax, canvas, data
    
    # グラフをクリア
    ax.cla()
    
    # データをプロット
    ax.plot(np.arange(len(data)), data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stage")
    ax.set_title("Sleep Stage Data")
    
    # グラフを更新
    canvas.draw()

# ファイル選択用の関数
def select_file():
    global canvas, data
    
    # ファイル選択ダイアログを表示
    file_path = askopenfilename(filetypes=[("Text Files", "*.txt")])
    
    # ファイルが選択された場合
    if file_path:
        # テキストファイルからデータを読み込む
        data = np.loadtxt(file_path)
        
        # データをプロット
        plot_data()
        
        # グラフをウィンドウに配置
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# 保存用の関数
def save_file():
    global data
    
    # ファイル保存ダイアログを表示
    file_path = asksaveasfilename(filetypes=[("Text Files", "*.txt")])
    
    # ファイルが選択された場合
    if file_path:
        # データを保存
        np.savetxt(file_path, data)
        
# 睡眠ステージ修正用の関数
def modify_stage():
    global data
    
    # 修正対象の範囲を入力
    start = input("Enter start time (sec): ")
    end = input("Enter end time (sec): ")
    new_stage = input("Enter new stage: ")
    
    # 入力値を数値に変換
    start = int(start)
    end = int(end)
    new_stage = int(new_stage)
    
    # 睡眠ステージを修正
    data[start:end] = new_stage
    
    # データを再プロット
    plot_data()

# GUIを作成
root = tk.Tk()
root.title("Sleep Stage Data Viewer")

# ファイル選択ボタンを作成
select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(side=tk.TOP, pady=10)

# ウィンドウを表示
root.mainloop()