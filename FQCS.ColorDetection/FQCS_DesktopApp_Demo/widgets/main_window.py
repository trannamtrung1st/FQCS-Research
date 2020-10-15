from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from views.main_window import Ui_MainWindow
from widgets.detection_cfg_screen import DetectionCfgScreen


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.showFullScreen()

        # screens
        self.screen_1 = DetectionCfgScreen()
        self.setCentralWidget(self.screen_1)
