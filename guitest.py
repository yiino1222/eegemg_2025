import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # プロット用のウィジェットを作成
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # ダミーデータを生成
        self.x = np.linspace(0, 10, 1000)
        self.y1 = np.sin(self.x)
        self.y2 = np.cos(self.x)

        # プロット用の曲線を作成
        self.curve1 = self.plot_widget.plot(self.x, self.y1, pen='r')
        self.curve2 = self.plot_widget.plot(self.x, self.y2, pen='b')

        # ツールバーを作成
        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)

        # ファイルオープンアクションを追加
        open_file_action = QtWidgets.QAction('Open', self)
        open_file_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(open_file_action)

    def open_file_dialog(self):
        # ファイルオープンダイアログを表示
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '',
                                                             'Data Files (*.csv *.txt);;All Files (*)',
                                                             options=options)

        # ファイルを読み込んでプロット
        if file_name:
            data = np.loadtxt(file_name, delimiter=',')
            self.curve1.setData(data[:, 0], data[:, 1])
            self.curve2.setData(data[:, 0], data[:, 2])


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())