from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import sys
from widgets.main_window import MainWindow
from models.counter import Counter
import threading


class ThreadJob(threading.Thread):
    def __init__(self, callback, event, interval):
        self.callback = callback
        self.event = event
        self.interval = interval
        super(ThreadJob, self).__init__()

    def run(self):
        while not self.event.wait(self.interval):
            self.callback()


def main():
    app = QApplication([])
    counter = Counter()

    def increase_counter():
        counter.count += 1
        counter.notify()
        print(counter.count)

    event = threading.Event()
    k = ThreadJob(increase_counter, event, 1)
    k.start()

    main_window = MainWindow()
    counter.register(main_window.counter_changed)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()