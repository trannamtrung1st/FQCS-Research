from PySide2.QtWidgets import *
from views.main_window import Ui_MainWindow
from PySide2.QtCore import *
from PySide2.QtGui import *
from models.data import Data


class MainWindow(QMainWindow):
    def __init__(self, data: Data):
        QMainWindow.__init__(self)
        # data
        self.data = data
        self.data.register(self.data_changed)

        # ui initialize
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.custom_ui()

    def custom_ui(self):
        self.showFullScreen()
        self.ui.btnQuit.clicked.connect(self.btn_quit_clicked)
        self.ui.btnAdd.clicked.connect(self.btn_add_data_clicked)
        self.data_changed(self.data)

    # events
    def btn_add_data_clicked(self):
        self.data.x += 1
        self.data.y += 1
        self.data.notify()

    def btn_quit_clicked(self):
        self.close()

    # data bindings
    def data_changed(self, data):
        x, y = data.x, data.y
        self.ui.lblData.setText(f"Data: {x},{y}")
