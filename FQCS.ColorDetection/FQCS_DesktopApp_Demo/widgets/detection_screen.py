from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from views.detection_screen import Ui_DetectionScreen


class DetectionScreen(QWidget):
    def __init__(self, on_capture_clicked: ()):
        QWidget.__init__(self)
        self.ui = Ui_DetectionScreen()
        self.ui.setupUi(self)

        # events
        self.ui.btnCapture.clicked.connect(on_capture_clicked)