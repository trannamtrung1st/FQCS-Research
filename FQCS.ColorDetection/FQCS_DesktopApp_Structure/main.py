import sys
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication
from widgets.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())