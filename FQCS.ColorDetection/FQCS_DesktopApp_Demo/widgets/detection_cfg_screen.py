from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from views.detection_cfg_screen import Ui_DetectionCfgScreen


class DetectionCfgScreen(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.ui = Ui_DetectionCfgScreen()
        self.ui.setupUi(self)
