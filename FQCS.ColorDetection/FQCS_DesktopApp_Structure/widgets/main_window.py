from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from views.main_window import Ui_MainWindow
from models.counter import Counter


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.clicked_count = 0
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btnQuit.clicked.connect(self.btn_quit_clicked)
        self.ui.btnAdd.clicked.connect(self.btn_add_clicked)

    # events handler
    def btn_quit_clicked(self):
        self.close()

    def btn_add_clicked(self):
        self.clicked_count += 1
        self.clicked_count_changed(self.clicked_count)

    # data bindings
    def counter_changed(self, counter: Counter):
        count = counter.count
        self.ui.lblCounter.setText(f"Counter: {count}")

    def clicked_count_changed(self, clicked_count: int):
        self.ui.lblClicked.setText(f"Clicked: {clicked_count}")