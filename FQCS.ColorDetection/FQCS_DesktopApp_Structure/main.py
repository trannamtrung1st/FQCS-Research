import sys
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication
from widgets.main_window import MainWindow
from models.data import Data

if __name__ == "__main__":
    app = QApplication([])
    data = Data()
    data.x, data.y = 1, 2
    main_window = MainWindow(data)
    main_window.show()
    sys.exit(app.exec_())